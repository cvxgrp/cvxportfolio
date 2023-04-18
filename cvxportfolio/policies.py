# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx

from cvxportfolio.costs import BaseCost
from cvxportfolio.returns import BaseReturnsModel
from cvxportfolio.constraints import BaseConstraint
from cvxportfolio.utils import values_in_time, null_checker
from .estimator import Estimator, DataEstimator
from .errors import MissingValuesError

__all__ = [
    "Hold",
    "SellAll",
    "FixedTrades",
    "PeriodicRebalance",
    "ProportionalRebalance",
    "AdaptiveRebalance",
    "SinglePeriodOpt",
    "MultiPeriodOpt",
    "ProportionalTradeToTargets",
    "RankAndLongShort",
    "FixedWeights"
]


class BaseTradingPolicy(Estimator):
    """Base class for a trading policy."""

    costs = []
    constraints = []

    ## TEMPORARY INTERFACE OLD NEW
    def get_trades(self, portfolio, t=dt.datetime.today()):
        """Trades list given current portfolio and time t."""
        value = sum(portfolio)
        w = portfolio / value
        return values_in_time(self, t, w) * value

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.0)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        return np.round(self.get_trades(portfolio, t) / values_in_time(prices, t))[:-1]


class Hold(BaseTradingPolicy):
    """Hold initial portfolio, don't trade."""

    def values_in_time(self, t, current_weights, *args, **kwargs):        
        """Update sub-estimators and produce current estimate."""
        return pd.Series(0.0, index=current_weights.index)


class RankAndLongShort(BaseTradingPolicy):
    """Rank assets by signal; long highest and short lowest.

    Args:
        signal (pd.DataFrame): time-indexed DataFrame of signal for all symbols
            excluding cash. At each point in time the num_long assets with
            highest signal will have equal positive weight, and the num_short assets with
            lower signal will have equal negative weight. If two or more assets have the same
            signal value and they are on the boundary of either the top or bottom set,
            alphanumerical ranking will prevail.
        num_long (int or pd.Series): number of assets to long, default 1; if specified as Series
            it must be indexed by time.
        num_short (int or pd.Series): number of assets to short, default 1; if specified as Series
            it must be indexed by time.
        target_leverage (float or pd.Series): leverage of the resulting portfolio, default 1;
            if specified as Series it must be indexed by time.
    """

    def __init__(self, signal, num_long=1, num_short=1, target_leverage=1.0):
        """Define sub-estimators at class attribute level."""
        self.num_long = DataEstimator(num_long)
        self.num_short = DataEstimator(num_short)
        self.signal = DataEstimator(signal)
        self.target_leverage = DataEstimator(target_leverage)

    def values_in_time(self, t, current_weights, *args, **kwargs):
        """Update sub-estimators and produce current estimate."""
        super().values_in_time(t, current_weights, *args, **kwargs)

        sorted_ret = pd.Series(
            self.signal.current_value, current_weights.index[:-1]
        ).sort_values()
        short_positions = sorted_ret.index[: self.num_short.current_value]
        long_positions = sorted_ret.index[-self.num_long.current_value :]

        target_weights = pd.Series(0.0, index=current_weights.index)
        target_weights[short_positions] = -1.0
        target_weights[long_positions] = 1.0

        target_weights /= sum(abs(target_weights))
        target_weights *= self.target_leverage.current_value

        # cash is always 1.
        target_weights[current_weights.index[-1]] = 1.0

        return target_weights - current_weights


class ProportionalTradeToTargets(BaseTradingPolicy):
    """Given a DataFrame of target weights in time, trade in equal proportions to reach those.

    Initially, it loads the list of trading days and so at each day it knows
    how many are missing before the next target day, and trades in equal proportions
    to reach those targets. If there are no targets remaining it defaults to
    not trading.
    
    Args:
        targets (pd.DataFrame): time-indexed DataFrame of target weight vectors at
            given points in time (e.g., start of each month). 
        
    """    
    def __init__(self, targets):
        self.targets = targets
        
    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Get list of trading days."""
        self.trading_days = returns.index
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)   
         
    def values_in_time(self, t, current_weights, *args, **kwargs):
        """Get current trade weights."""
        super().values_in_time(t, current_weights, *args, **kwargs) 
        next_targets = self.targets.loc[self.targets.index > t]
        if not len(next_targets):
            return pd.Series(0.0, index=current_weights.index)
        next_target = next_targets.iloc[0]
        next_target_day = next_targets.index[0]
        trading_days_to_target = len(self.trading_days[(self.trading_days>=t) & 
            (self.trading_days<next_target_day)])
        return (next_target - current_weights) / trading_days_to_target
        


class SellAll(BaseTradingPolicy):
    """Sell all assets to cash.
    
    This is useful to check the tcost model in the simulator,
    or as an element in a (currently not implemented) composite policy.
    """
    
    def values_in_time(self, t, current_weights, *args, **kwargs):
        """Get current trade weights."""
        super().values_in_time(t, current_weights, *args, **kwargs)
        target = np.zeros(len(current_weights))
        target[-1] = 1.
        return target - current_weights


class FixedTrades(BaseTradingPolicy):
    """Each day trade the provided trade weights vector.
    
    If there are no weights defined for the given day, default to no 
    trades.
    
    Args:
        trades_weights (pd.Series or pd.DataFrame): Series of weights 
            (if constant in time) or DataFrame of trade weights
            indexed by time. It trades each day the corresponding vector.
    """

    def __init__(self, trades_weights):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        self.trades_weights = DataEstimator(trades_weights)

    def values_in_time(self, t, current_weights, *args, **kwargs):
        try:
            super().values_in_time(t, current_weights, *args, **kwargs)
            return pd.Series(self.trades_weights.current_value, current_weights.index)
        except MissingValuesError:
            return pd.Series(0., current_weights.index)
        
        
class FixedWeights(BaseTradingPolicy):
    """Each day trade to the provided trade weights vector.
    
    If there are no weights defined for the given day, default to no 
    trades.
    
    Args:
        target_weights (pd.Series or pd.DataFrame): Series of weights 
            (if constant in time) or DataFrame of trade weights
            indexed by time. It trades each day to the corresponding vector.
    """

    def __init__(self, target_weights):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        self.target_weights = DataEstimator(target_weights)

    def values_in_time(self, t, current_weights, *args, **kwargs):
        try:
            super().values_in_time(t, current_weights, *args, **kwargs)
            return pd.Series(self.target_weights.current_value, current_weights.index) - current_weights
        except MissingValuesError:
            return pd.Series(0., current_weights.index)
            
        
class PeriodicRebalance(FixedWeights):
    """Track a target weight vector rebalancing at given times.
    
    This calls `FixedWeights`. If you want to change the target in time
    use that policy directly.


    Args:
        target (pd.Series): portfolio weights to rebalance to.
        rebalancing_times (pd.DateTimeIndex): after the open trading on these days
            portfolio is equal to target.
    """

    def __init__(self, target, rebalancing_times):
        target_weights = pd.DataFrame({el:target for el in rebalancing_times}).T
        super().__init__(target_weights)


class ProportionalRebalance(ProportionalTradeToTargets):
    """Track a target weight exactly at given times, trading proportionally to it each period.
    
    This calls `ProportionalTradeToTargets`. If you want to change the target in time
    use that policy directly.

    Args:
        target (pd.Series): portfolio weights to rebalance to.
        target_matching_times (pd.DateTimeIndex): after the open trading on these days
            portfolio is equal to target.
    """
    def __init__(self, target, target_matching_times):
        targets = pd.DataFrame({el:target for el in target_matching_times}).T
        super().__init__(targets)


class AdaptiveRebalance(BaseTradingPolicy):
    """Rebalance portfolio when deviates too far from target.
    
    We use the 2-norm as trigger for rebalance. You may want to
    calibrate the `max_tracking_error` for your application 
    by backtesting this policy, e.g., to get your desired turnover.
    
    Args:
        target (pd.Series or pd.DataFrame): target weights to rebalance to.
            It is assumed a constant if it is a Series. If it varies in 
            time (you must specify it for every trading day) pass a 
            DataFrame indexed by time.
        tracking_error (float or pd.Series): we trade to match the target
            weights whenever the 2-norm of our weights minus the
            target is larger than this. Pass a Series if you want to vary it in
            time.
    """

    def __init__(self, target, tracking_error):
        self.target = DataEstimator(target)
        self.tracking_error = DataEstimator(tracking_error)
        
    def values_in_time(self, t, current_weights, *args, **kwargs):
        super().values_in_time(t, current_weights, *args, **kwargs)
        if np.linalg.norm(current_weights - self.target.current_value) > self.tracking_error.current_value:
            return self.target.current_value - current_weights
        else:
            return pd.Series(0., current_weights.index)
        

class SinglePeriodOpt(BaseTradingPolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(
        self, return_forecast, costs, constraints, solver=None, solver_opts=None
    ):
        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast

        super(SinglePeriodOpt, self).__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(
                cvx.multiply(values_in_time(self.return_forecast, t).values, wplus)
            )

        assert alpha_term.is_concave()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        for constr in self.constraints:
            constraints += constr.weight_expr(t, wplus, z, value)

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        self.prob = cvx.Problem(
            cvx.Maximize(alpha_term - sum(costs)), [cvx.sum(z) == 0] + constraints
        )
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except (cvx.SolverError, TypeError):
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(portfolio)


# class LookaheadModel():
#     """Returns the planning periods for multi-period.
#     """
#     def __init__(self, trading_times, period_lens):
#         self.trading_times = trading_times
#         self.period_lens = period_lens
#
#     def get_periods(self, t):
#         """Returns planning periods.
#         """
#         periods = []
#         tau = t
#         for length in self.period_lens:
#             incr = length*pd.Timedelta('1 days')
#             periods.append((tau, tau + incr))
#             tau += incr
#         return periods


class MultiPeriodOpt(SinglePeriodOpt):
    def __init__(
        self, trading_times, terminal_weights, lookahead_periods=None, *args, **kwargs
    ):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        # Should there be a constraint that the final portfolio is the bmark?
        self.terminal_weights = terminal_weights
        super(MultiPeriodOpt, self).__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):
        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.get_loc(t) : self.trading_times.get_loc(t)
            + self.lookahead_periods
        ]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:

            #            tau = t + delta_t
            z = cvx.Variable(*w.shape)
            wplus = w + z
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(z) == 0]
            for single_constr in self.constraints:
                constr += single_constr.weight_expr(t, wplus, z, value)

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            # prob_arr[-1].constraints += [wplus == self.terminal_weights.values]
            prob_arr[-1] = cvx.Problem(
                cvx.Maximize(obj), constr + [wplus == self.terminal_weights.values]
            )

        sum(prob_arr).solve(solver=self.solver)
        return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))
