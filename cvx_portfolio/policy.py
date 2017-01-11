"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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

from .costs import BaseCost
from .returns import BaseAlphaModel
from .constraints import BaseConstraint


__all__ = ['Hold', 'PeriodicRebalance', 'AdaptiveRebalance']


class BasePolicy():
    """ Base class for a trading policy. """
    __metaclass__ = ABCMeta

    costs = []
    constraints = []

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def get_trades(self, portfolio, t):
        """Trades list given current portfolio and time t.
        """
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

class Hold(BasePolicy):
    """Hold initial portfolio.
    """
    def get_trades(self, portfolio, t):
        return self._nulltrade(portfolio)


class BaseRebalance(BasePolicy):

    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class PeriodicRebalance(BaseRebalance):
    """Track a target portfolio, rebalancing at given times.
    """
    def __init__(self, target, rebalancing_times, name="PeriodicRebalance"):
        """

        Args:
            target: target weights, n+1 vector
            rebalancing_times: iterable/set of datetime objects, times at which we want to rebalance
        """
        self.target = target
        self.rebalancing_times = rebalancing_times
        self.name = name

    def get_trades(self, portfolio, t):
        if t in self.rebalancing_times:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class AdaptiveRebalance(BaseRebalance):
    """ Rebalance portfolio when deviates too far from target.
    """
    def __init__(self, target, tracking_error):
        self.target = target
        self.tracking_error = tracking_error

    def get_trades(self, portfolio, t):
        weights=portfolio/sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class SinglePeriodOpt(BasePolicy):

    def __init__(self, alpha_model, costs, constraints, solver=None, name="SinglePeriodOpt"):

        self.alpha_model = alpha_model
        assert isinstance(self.alpha_model, BaseAlphaModel)

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.name = name

    def get_trades(self, portfolio, t):

        value = sum(portfolio)
        w = portfolio/value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z
        wbench = np.zeros(len(w)) # TODO FIX THIS ##portfolio.benchmark.values

        alpha_term = self.alpha_model.weight_expr(t, wplus)
        assert(alpha_term.is_concave())

        costs = [cost.weight_expr(t, wplus, wbench, z, value) for cost in self.costs]
        for el in costs:
            assert (el.is_convex())

        constraints = [item for item in (con.weight_expr(t, wplus, wbench, z, value) for con in self.constraints)]
        for el in constraints:
            assert (el.is_dcp())

        prob = cvx.Problem(
            cvx.Maximize(alpha_term - sum(costs)),
            [cvx.sum_entries(z) == 0] + constraints)
        try:
            prob.solve(solver=self.solver, verbose=False)

            if prob.status == 'unbounded':
                logging.error('The problem is unbounded. Defaulting to no trades')
                return self._nulltrade(portfolio)

            if prob.status == 'infeasible':
                logging.error('The problem is infeasible. Defaulting to no trades')
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value.A1 * value))  # TODO will have index
        except cvx.SolverError:
            logging.error('The solver %s failed. Defaulting to no trades' % self.solver)
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
#             incr = length*pd.Timedelta('1 days')  # TODO generalize to non-days
#             periods.append((tau, tau + incr))
#             tau += incr
#         return periods


class MultiPeriodOpt(SinglePeriodOpt):

    def __init__(self, lookahead_periods, alpha_model, costs, constraints,
                 terminal_constr=True, solver=None, name="MultiPeriodOpt"):
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        # Should there be a constraint that the final portfolio is the bmark?
        self.terminal_constr = terminal_constr
        super(MultiPeriodOpt, self).__init__(alpha_model, costs, constraints, solver, name)

    def get_trades(self, portfolio, t):

        value = portfolio.v
        # value must be positive.
        assert (value > 0.)
        # if value <= 0:
        #     return self._nulltrade(portfolio)

        w = portfolio.w.values
        prob_arr = []
        z_vars = []
        wbench = portfolio.benchmark.values

        # planning_periods = self.lookahead_model.get_periods(t)
        for delta_t in [pd.Timedelta('%d days' % i) for i in range(self.lookahead_periods)]:

            tau = t + delta_t
            z = cvx.Variable(portfolio.w.size)
            wplus = w + z
            obj = self.alpha_model.weight_expr_ahead(t, tau, wplus)
            obj -= sum([cost.weight_expr_ahead(t, tau, wplus, wbench, z, value) for cost in self.costs])
            constr = [cvx.sum_entries(z) == 0]
            constr += [con.weight_expr(t, wplus, wbench, z, value) for con in self.constraints]  # TODO should be est_ahead
            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_constr:
            prob_arr[-1].constraints += [wplus == wbench]

        sum(prob_arr).solve(solver=self.solver)
        return pd.Series(index=portfolio.h.index, data=(z_vars[0].value.A1 * value))
