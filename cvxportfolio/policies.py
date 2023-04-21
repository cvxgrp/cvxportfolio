# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
# Copyright 2023- The Cvxportfolio Contributors
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
"""This module contains trading policies that can be backtested."""

import datetime as dt
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx

from .costs import BaseCost
from .returns import BaseReturnsModel
from .constraints import BaseConstraint
#from cvxportfolio.utils import values_in_time, null_checker
from .estimator import Estimator, DataEstimator
from .errors import MissingValuesError, PortfolioOptimizationError

__all__ = [
    "Hold",
    "SellAll",
    "FixedTrades",
    "PeriodicRebalance",
    "ProportionalRebalance",
    "AdaptiveRebalance",
    "SinglePeriodOptimization",
    "MultiPeriodOptimization",
    "ProportionalTradeToTargets",
    "RankAndLongShort",
    "FixedWeights",
]


class BaseTradingPolicy(Estimator):
    """Base class for a trading policy."""

    costs = []
    constraints = []
    
    INITIALIZED = False # used to interface w/ old cvxportfolio

    # TEMPORARY INTERFACE OLD NEW
    def get_trades(self, portfolio, t=dt.datetime.today()):
        """Trades list given current portfolio and time t."""
        value = sum(portfolio)
        w = portfolio / value
        if not self.INITIALIZED:
            self.pre_evaluation(returns=pd.DataFrame(0.0, index=[t], columns=portfolio.index),
                volumes=None, start_time=t, end_time=None)
            self.INITIALIZED = True
        return self.values_in_time(t, current_weights=w,
                current_portfolio_value=value, past_returns=None, past_volumes=None) * value

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.0)


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
        long_positions = sorted_ret.index[-self.num_long.current_value:]

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
    how many are missing before the next target's day, and trades in equal proportions
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
        trading_days_to_target = len(self.trading_days[(
            self.trading_days >= t) & (self.trading_days < next_target_day)])
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
        target[-1] = 1.0
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
            return pd.Series(
                self.trades_weights.current_value,
                current_weights.index)
        except MissingValuesError:
            return pd.Series(0.0, current_weights.index)


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
            return (
                pd.Series(
                    self.target_weights.current_value,
                    current_weights.index) -
                current_weights)
        except MissingValuesError:
            return pd.Series(0.0, current_weights.index)


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
        target_weights = pd.DataFrame(
            {el: target for el in rebalancing_times}).T
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
        targets = pd.DataFrame({el: target for el in target_matching_times}).T
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
        if (
            np.linalg.norm(current_weights - self.target.current_value)
            > self.tracking_error.current_value
        ):
            return self.target.current_value - current_weights
        else:
            return pd.Series(0.0, current_weights.index)



class MultiPeriodOptimization(BaseTradingPolicy):
    """Multi Period Optimization policy.

    Implements the model developed in Chapter 5, in particular
    at page 49, of the book. You specify the objective terms
    using classes such as ReturnsForecast and TcostModel, each
    multiplied by its multiplier. You also specify lists
    of constraints. There are two ways to do it. You either
    define the same objective terms and costraints for each 
    step of the multi-period problem, or you define a different
    objective term and different list of constraints for each step.
    In addition we offer a `terminal_constraint` argument to
    simply impose that at the last step in the optimization the
    post-trade weights match the given weights (see page 51).
    
    When it computes the trajectory of weights for the future
    it only returns the first step (to the Simulator, typically).
    The future steps (planning horizon) are by default not returned.

    Args:
        objective (algebra of BaseCost or list of those): these will be maximized;
            if you pass a single expression of BaseCost it is understood as the 
            same for all steps; if it's a list you must also pass a list of lists
            for `constraints`, each term represents the cost for each step of the optimization
            (starting from the first, i.e., today) and the length of the list is 
            used as planning_horizon (the value you pass there will be ignored) 
        constraints (list of BaseConstraints or list of those): these will be
            imposed on the optimization. Default []. Pass this as a list of
            lists of the same length as `objective` to specify different 
            constraints at different time steps.
        planning_horizon (int or None): how many steps in the future we 
            plan for. Ignored if passing `objective` and `constraints` as lists.
            Default is None.
        terminal_constraint (pd.Series or None): if you pass a Series to this
            (default is None) it will impose that at the last step of the multi
            period optimization the post-trade weights are equal to this.
        **kwargs: these will be passed to cvxpy.Problem.solve,
            so you can choose your own solver and pass
            parameters to it.
    """

    def __init__(self, objective, constraints=[], planning_horizon=None,terminal_constraint=None,**kwargs):
        if hasattr(objective, '__iter__'):
            if not (hasattr(constraints, '__iter__') and len(constraints) and (hasattr(constraints[0], '__iter__') and len(objective) == len(constraints))):
                raise SyntaxError('If you pass objective as a list, constraints should be a list of lists of the same length.')
            self.planning_horizon = len(objective)
            self.objective = objective
            self.constraints = constraints
        else:
            if not np.isscalar(planning_horizon):
                raise SyntaxError('If `objective` and `constraints` are the same for all steps you must specify `planning_horizon`.')
            self.planning_horizon = planning_horizon
            self.objective = [objective for i in range(planning_horizon)]
            self.constraints = [constraints for i in range(planning_horizon)]
                
        self.terminal_constraint = terminal_constraint
        self.cvxpy_kwargs = kwargs

    def compile_to_cvxpy(self):#, w_plus, z, value):
        """Compile all cvxpy expressions and the problem."""
        self.cvxpy_objective = [
            el.compile_to_cvxpy(self.w_plus_at_lags[i], self.z_at_lags[i], self.portfolio_value) 
            for i, el in enumerate(self.objective)]
        self.cvxpy_objective = sum(self.cvxpy_objective)
        assert self.cvxpy_objective.is_dcp()  # dpp=True)
        assert self.cvxpy_objective.is_concave()
        self.cvxpy_constraints = [
            [constr.compile_to_cvxpy(self.w_plus_at_lags[i], self.z_at_lags[i], self.portfolio_value) 
                for constr in el]
            for i, el in enumerate(self.constraints)]
        self.cvxpy_constraints = sum(self.cvxpy_constraints, [])
        self.cvxpy_constraints += [cvx.sum(z) == 0 for z in self.z_at_lags]
        w = self.w_current
        for i in range(self.planning_horizon):
            self.cvxpy_constraints.append(self.w_plus_at_lags[i] == self.z_at_lags[i] + w)
            w = self.w_plus_at_lags[i]
        if not self.terminal_constraint is None:
            self.cvxpy_constraints.append(w == self.terminal_constraint)
        self.problem = cvx.Problem(cvx.Maximize(self.cvxpy_objective), self.cvxpy_constraints)
        assert self.problem.is_dcp()  # dpp=True)

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Pass a full view of the data to initialize objects that need it."""
        for obj in self.objective:
            obj.pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        for constr_at_lag in self.constraints:
            for constr in constr_at_lag:
                constr.pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

        # initialize the problem
        self.portfolio_value = cvx.Parameter(nonneg=True)
        self.w_current = cvx.Parameter(returns.shape[1])
        self.z_at_lags = [cvx.Variable(returns.shape[1]) for i in range(self.planning_horizon)] 
        self.w_plus_at_lags = [cvx.Variable(returns.shape[1]) for i in range(self.planning_horizon)]

        self.compile_to_cvxpy()#self.w_plus, self.z, self.portfolio_value)

    def values_in_time(self, t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs):
        """Update all cvxpy parameters and solve."""
        for obj in self.objective:
            obj.values_in_time(t, current_weights, current_portfolio_value,
                past_returns, past_volumes, **kwargs)
        for constr_at_lag in self.constraints:
            for constr in constr_at_lag:
                constr.values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)

        self.portfolio_value.value = current_portfolio_value
        self.w_current.value = current_weights.values
        try:
            self.problem.solve(**self.cvxpy_kwargs)
        except cvx.SolverError:
            raise PortfolioOptimizationError(
                f"Numerical solver for policy {self.__class__.__name__} at time {t} failed;"
                "try changing it, relaxing some constraints, or dropping some costs.")
        if self.problem.status in ["unbounded", "unbounded_inaccurate"]:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time {t} resulted in an unbounded problem."
            )
        if self.problem.status in ["infeasible", 'infeasible_inaccurate']:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time {t} resulted in an infeasible problem."
            )

        return pd.Series(self.z_at_lags[0].value, current_weights.index)


class SinglePeriodOptimization(MultiPeriodOptimization):
    """Single Period Optimization policy.

    Implements the model developed in Chapter 4, in particular
    at page 43, of the book. You specify the objective term
    using classes such as ReturnsForecast and TcostModel, each
    multiplied by its multiplier. You also specify a list
    of constraints.

    Args:
        objective (algebra of BaseCost): this will be maximized.
        constraints (list of BaseConstraints): these will be
            imposed on the optimization. Default [].
        **kwargs: these will be passed to cvxpy.Problem.solve,
            so you can choose your own solver and pass
            parameters to it.
    """

    def __init__(self, objective, constraints=[], **kwargs):
        super().__init__([objective], [constraints], **kwargs)
        

class SinglePeriodOptOLDTONEW(SinglePeriodOptimization):
    """Placeholder class while we translate tests to new interface."""

    def __init__(
        self, return_forecast, costs, constraints, solver=None, solver_opts={}
    ):
        if np.isscalar(return_forecast):
            raise Exception
        if hasattr(return_forecast, "index"):
            return_forecast = ReturnsForecast(return_forecast)
        objective = -sum(costs, start=-return_forecast)
        kwargs = solver_opts
        if not (solver is None):
            kwargs["solver"] = solver
        super().__init__(objective, constraints, **kwargs)
