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

import logging

import cvxpy as cvx
import pandas as pd

from cvx_portfolio.costs import BaseCost
from cvx_portfolio.returns import BaseAlphaModel
from .base_policy import BasePolicy
from .constraints import BaseConstraint


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

        value = portfolio.v
        w = portfolio.w
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z
        wbench = portfolio.benchmark.values

        alpha_term = self.alpha_model.weight_expr(t, wplus)
        assert(alpha_term.is_concave())

        # TODO fix the [:-1]
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
            return pd.Series(index=portfolio.h.index, data=(z.value.A1 * value))  # TODO will have index
        except cvx.SolverError:
            logging.ERROR('The solver %s failed. Defaulting to no trades' % self.solver)
            return self._nulltrade(portfolio)
