"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx

from cvxportfolio.costs import BaseCost
from cvxportfolio.returns import BaseReturnsModel
from cvxportfolio.constraints import BaseConstraint
from cvxportfolio.utils import values_in_time, null_checker


__all__ = ['Hold', 'FixedTrade', 'PeriodicRebalance', 'AdaptiveRebalance',
           'SinglePeriodOpt', 'MultiPeriodOpt', 'ProportionalTrade',
           'RankAndLongShort']


class BasePolicy(object, metaclass=ABCMeta):
    """ Base class for a trading policy. """

    def __init__(self):
        self.costs = []
        self.constraints = []

    @abstractmethod
    def get_trades(self, portfolio, t=pd.datetime.today()):
        """Trades list given current portfolio and time t.
        """
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        return np.round(self.get_trades(portfolio,
                                        t) / values_in_time(prices, t))[:-1]


class Hold(BasePolicy):
    """Hold initial portfolio.
    """

    def get_trades(self, portfolio, t=pd.datetime.today()):
        return self._nulltrade(portfolio)


class RankAndLongShort(BasePolicy):
    """Rank assets, long the best and short the worst (cash neutral)."""

    def __init__(self, return_forecast, num_long, num_short, target_turnover):
        self.target_turnover = target_turnover
        self.num_long = num_long
        self.num_short = num_short
        self.return_forecast = return_forecast
        super(RankAndLongShort, self).__init__()

    def get_trades(self, portfolio, t=pd.datetime.today()):
        prediction = values_in_time(self.return_forecast, t)
        sorted_ret = prediction.sort_values()

        short_trades = sorted_ret.index[:self.num_short]
        long_trades = sorted_ret.index[-self.num_long:]

        u = pd.Series(0., index=prediction.index)
        u[short_trades] = -1.
        u[long_trades] = 1.
        u /= sum(abs(u))
        u = sum(portfolio) * u * self.target_turnover

        # import pdb; pdb.set_trace()
        #
        # # ex-post cash neutrality
        # old_cash = portfolio[-1]
        # if old_cash > 0:
        #     u[short] = u[short] + old_cash/self.num_short
        # else:
        #     u[long] = u[long] + old_cash/self.num_long

        return u


class ProportionalTrade(BasePolicy):
    """Gets to target in given time steps."""

    def __init__(self, targetweight, time_steps):
        self.targetweight = targetweight
        self.time_steps = time_steps
        super(ProportionalTrade, self).__init__()

    def get_trades(self, portfolio, t=pd.datetime.today()):
        try:
            missing_time_steps = len(
                self.time_steps) - next(i for (i, x)
                                        in enumerate(self.time_steps)
                                        if x == t)
        except StopIteration:
            raise Exception(
                "ProportionalTrade can only trade on the given time steps")
        deviation = self.targetweight - portfolio / sum(portfolio)
        return sum(portfolio) * deviation / missing_time_steps


class SellAll(BasePolicy):
    """Sell all non-cash assets."""

    def get_trades(self, portfolio, t=pd.datetime.today()):
        trade = -pd.Series(portfolio, copy=True)
        trade.ix[-1] = 0.
        return trade


class FixedTrade(BasePolicy):
    """Trade a fixed trade vector.
    """

    def __init__(self, tradevec=None, tradeweight=None):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        if tradevec is not None and tradeweight is not None:
            raise Exception
        if tradevec is None and tradeweight is None:
            raise Exception
        self.tradevec = tradevec
        self.tradeweight = tradeweight
        assert(self.tradevec is None or sum(self.tradevec) == 0.)
        assert(self.tradeweight is None or sum(self.tradeweight) == 0.)
        super(FixedTrade, self).__init__()

    def get_trades(self, portfolio, t=pd.datetime.today()):
        if self.tradevec is not None:
            return self.tradevec
        return sum(portfolio) * self.tradeweight


class BaseRebalance(BasePolicy):

    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class PeriodicRebalance(BaseRebalance):
    """Track a target portfolio, rebalancing at given times.
    """

    def __init__(self, target, period, **kwargs):
        """
        Args:
            target: target weights, n+1 vector
            period: supported options are "day", "week", "month", "quarter",
                "year".
                rebalance on the first day of each new period
        """
        self.target = target
        self.period = period
        super(PeriodicRebalance, self).__init__()

    def is_start_period(self, t):
        result = not getattr(t, self.period) == getattr(self.last_t,
                                                        self.period) if \
            hasattr(self,
                    'last_t')\
            else True
        self.last_t = t
        return result

    def get_trades(self, portfolio, t=pd.datetime.today()):
        return self._rebalance(portfolio) if self.is_start_period(t) else \
            self._nulltrade(portfolio)


class AdaptiveRebalance(BaseRebalance):
    """ Rebalance portfolio when deviates too far from target.
    """

    def __init__(self, target, tracking_error):
        self.target = target
        self.tracking_error = tracking_error
        super(AdaptiveRebalance, self).__init__()

    def get_trades(self, portfolio, t=pd.datetime.today()):
        weights = portfolio / sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class SinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(self, return_forecast, costs, constraints, solver=None,
                 solver_opts=None):

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
            t = pd.datetime.today()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(cvx.multiply(
                values_in_time(self.return_forecast, t).values,
                wplus))

        assert(alpha_term.is_concave())

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [item for item in (con.weight_expr(t, wplus, z, value)
                                          for con in self.constraints)]

        for el in costs:
            assert (el.is_convex())

        for el in constraints:
            assert (el.is_dcp())

        self.prob = cvx.Problem(
            cvx.Maximize(alpha_term - sum(costs)),
            [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == 'unbounded':
                logging.error(
                    'The problem is unbounded. Defaulting to no trades')
                return self._nulltrade(portfolio)

            if self.prob.status == 'infeasible':
                logging.error(
                    'The problem is infeasible. Defaulting to no trades')
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except cvx.SolverError:
            logging.error(
                'The solver %s failed. Defaulting to no trades' % self.solver)
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

    def __init__(self, trading_times,
                 terminal_weights, lookahead_periods=None, *args, **kwargs):
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

    def get_trades(self, portfolio, t=pd.datetime.today()):

        value = sum(portfolio)
        assert (value > 0.)
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in \
                self.trading_times[self.trading_times.index(t):
                                   self.trading_times.index(t) +
                                   self.lookahead_periods]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:

            #            tau = t + delta_t
            z = cvx.Variable(*w.shape)
            wplus = w + z
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(
                    t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(z) == 0]
            constr += [con.weight_expr(t, wplus, z, value)
                       for con in self.constraints]

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        sum(prob_arr).solve(solver=self.solver)
        return pd.Series(index=portfolio.index,
                         data=(z_vars[0].value * value))
