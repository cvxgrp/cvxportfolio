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


import cvxpy as cvx
import pandas as pd
from .single_period import SinglePeriodOpt


# TODO implement constraints (dynamic, modular)

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
