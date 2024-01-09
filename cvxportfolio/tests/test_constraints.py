# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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
"""Unit tests for the constraints objects."""

import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.tests import CvxportfolioTest

VALUES_IN_TIME_DUMMY_KWARGS = {
    'current_weights': None,
    'past_returns': None,
    'current_prices': None,
    'past_volumes': None
}

class TestConstraints(CvxportfolioTest):
    """Test Cvxportfolio constraint objects."""

    def _build_constraint(self, constraint, t=None):
        """Initialize constraint, build expression, and point it to time."""
        constraint.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        cvxpy_expression = constraint.compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        constraint.values_in_time_recursive(
            t=pd.Timestamp("2020-01-01") if t is None else t,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS
            )
        return cvxpy_expression

    def test_long_only(self):
        """Test long-only constraint."""
        model = cvx.LongOnly()
        cons = self._build_constraint(model)
        self.w_plus.value = np.ones(self.N)
        self.assertTrue(cons.value())
        self.w_plus.value = -np.ones(self.N)
        self.assertFalse(cons.value())

    def test_nocash(self):
        """Test no cash constraint."""
        model = cvx.NoCash()
        cons = self._build_constraint(model)
        self.w_plus.value = np.ones(self.N)
        self.assertFalse(cons.value())
        self.w_plus.value = -np.ones(self.N)
        self.assertFalse(cons.value())
        self.w_plus.value[-1] = 0.
        self.assertTrue(cons.value())

    def test_long_cash(self):
        """Test long-cash constraint."""
        model = cvx.LongCash()
        cons = self._build_constraint(model)
        self.w_plus.value = np.ones(self.N)
        self.assertTrue(cons.value())
        tmp = np.ones(self.N)
        tmp[-1] = -1
        self.w_plus.value = tmp
        self.assertTrue(cons.is_dcp())
        self.assertFalse(cons.value())

    def test_min_cash(self):
        """Test min-cash constraint."""
        model = cvx.MinCashBalance(10000)  # USD
        cons = self._build_constraint(model)
        self.w_plus.value = np.zeros(self.N)
        self.w_plus.value[-1] = 1
        model.values_in_time_recursive(t=pd.Timestamp(
            "2020-01-01"), current_portfolio_value=10001,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=pd.Timestamp(
            "2020-01-01"), current_portfolio_value=9999,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_dollar_neutral(self):
        """Test dollar-neutral constraint."""
        model = cvx.DollarNeutral()
        cons = self._build_constraint(model)
        tmpvalue = np.zeros(self.N)
        tmpvalue[-1] = 1 - sum(tmpvalue[:-1])
        self.w_plus.value = tmpvalue
        self.assertTrue(cons.value())
        tmpvalue = np.ones(self.N)
        tmpvalue[-1] = 1 - sum(tmpvalue[:-1])
        self.w_plus.value = tmpvalue
        self.assertFalse(cons.value())

    def test_leverage_limit(self):
        """Test leverage limit constraint."""
        model = cvx.LeverageLimit(2)
        cons = self._build_constraint(model)
        self.w_plus.value = np.ones(self.N) / self.N
        self.assertTrue(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        model = cvx.LeverageLimit(7)
        cons = self._build_constraint(model)
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_leverage_limit_in_time(self):
        """Test leverage limit constraint with time-varying limit."""
        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = 7
        model = cvx.LeverageLimit(limits)
        cons = self._build_constraint(model, t=self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=self.returns.index[2],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_max_weights(self):
        """Test max weights constraint."""
        model = cvx.MaxWeights(2)
        cons = self._build_constraint(model)
        self.w_plus.value = np.ones(self.N) / self.N
        self.assertTrue(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())

        model = cvx.MaxWeights(7)
        cons = self._build_constraint(model)

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = 7

        model = cvx.MaxWeights(limits)
        cons = self._build_constraint(model, t=self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=self.returns.index[2],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_min_weights(self):
        """Test min weights constraint."""
        model = cvx.MinWeights(2)
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        model = cvx.MinWeights(-3)
        cons = self._build_constraint(model, self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = -3
        model = cvx.MinWeights(limits)
        cons = self._build_constraint(model, t=self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=self.returns.index[2],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_max_bm_dev(self):
        """Test max benchmark deviation constraint."""
        model = cvx.MaxBenchmarkDeviation(2)
        cons = self._build_constraint(model)
        self.w_plus_minus_w_bm.value = np.ones(self.N) / self.N
        self.assertTrue(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertFalse(cons.value())

        model = cvx.MaxBenchmarkDeviation(7)
        cons = self._build_constraint(model)

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = 7

        model = cvx.MaxBenchmarkDeviation(limits)
        cons = self._build_constraint(model, t=self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=self.returns.index[2],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_min_bm_dev(self):
        """Test min benchmark deviation constraint."""
        model = cvx.MinBenchmarkDeviation(2)
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus_minus_w_bm.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertFalse(cons.value())
        model = cvx.MinBenchmarkDeviation(-3)
        cons = self._build_constraint(model, self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = -3
        model = cvx.MinBenchmarkDeviation(limits)
        cons = self._build_constraint(model, t=self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus_minus_w_bm.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time_recursive(t=self.returns.index[2],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertFalse(cons.value())

    def test_factor_max_limit(self):
        """Test factor max limit constraint."""

        model = cvx.FactorMaxLimit(
            np.ones((self.N - 1, 2)), np.array([0.5, 1]))
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())

        model = cvx.FactorMaxLimit(np.ones((self.N - 1, 2)), np.array([4, 4]))
        cons = self._build_constraint(model, self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_factor_min_limit(self):
        """Test factor min limit constraint."""

        model = cvx.FactorMinLimit(
            np.ones((self.N - 1, 2)), np.array([0.5, 1]))
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3

        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        model = cvx.FactorMinLimit(np.ones((self.N - 1, 2)), np.array([4, 4]))
        cons = self._build_constraint(model, self.returns.index[1])
        # cons = model.weight_expr(t, self.w_plus, None, None)[0]
        tmp = np.zeros(self.N)
        tmp[0] = 4
        # tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_factor_gross_limit(self):
        """Test factor gross limit constraint."""

        model = cvx.FactorGrossLimit(
            np.ones((self.N - 1, 2)), np.array([0.5, 1]))
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())

        model = cvx.FactorGrossLimit(
            np.ones((self.N - 1, 2)), np.array([7, 7]))
        cons = self._build_constraint(model, self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_fixed_alpha(self):
        """Test fixed alpha constraint."""

        model = cvx.FixedFactorLoading(np.ones((self.N - 1, 1)), 1)
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_factor_neutral(self):
        """Test FactorNeutral constraint."""

        model = cvx.FactorNeutral(np.ones((self.N - 1, 1)))
        cons = self._build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -4
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_turnover_limit(self):
        """Test turnover limit constraint."""
        model = cvx.TurnoverLimit(0.1)
        cons = self._build_constraint(model)
        self.z.value = np.zeros(self.N)
        self.z.value[-1] = -sum(self.z.value[:-1])
        self.assertTrue(cons.value())

        self.z.value[1] = 0.2
        self.z.value[-1] = -sum(self.z.value[:-1])
        self.assertTrue(cons.value())

        self.z.value[2] = -0.01
        self.z.value[-1] = -sum(self.z.value[:-1])
        self.assertFalse(cons.value())

    def test_participation_rate(self):
        """Test max participation rate constraints."""

        t = self.returns.index[1]

        # avg daily value limits.
        value = 1e6
        model = cvx.ParticipationRateLimit(
            self.volumes, max_fraction_of_volumes=0.1)
        model.initialize_estimator_recursive(
            universe=self.returns.columns,
            trading_calendar=self.returns.index)
        cons = model.compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        model.values_in_time_recursive(t=t, current_portfolio_value=value)
        print(model.portfolio_value.value)
        # cons = model.weight_expr(t, None, z, value)[0]
        tmp = np.zeros(self.N)
        tmp[:-1] = self.volumes.loc[t].values / value * 0.05
        self.z.value = tmp
        self.assertTrue(cons.value())
        self.z.value = -100 * self.z.value  # -100*np.ones(n)
        self.assertFalse(cons.value())


if __name__ == '__main__':

    unittest.main() # pragma: no cover
