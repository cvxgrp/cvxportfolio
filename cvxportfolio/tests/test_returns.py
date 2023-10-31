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
"""Unit tests for the return forecast objects."""

import unittest

import numpy as np

from cvxportfolio.returns import (CashReturn, ReturnsForecast,
                                  ReturnsForecastError)
from cvxportfolio.tests import CvxportfolioTest


class TestReturns(CvxportfolioTest):
    """Test return objects."""

    def test_cash_returns(self):
        """Test CashReturn object with last return as forecast."""
        cash_model = CashReturn()
        cvxpy_expression = self.boilerplate(cash_model)
        self.w_plus.value = np.random.randn(self.N)
        cash_model.values_in_time_recursive(
            t=None, past_returns=self.returns.iloc[:123])
        cr = self.returns.iloc[122, -1]
        self.assertTrue(cvxpy_expression.value == cr * (
            self.w_plus[-1].value ))

    def test_cash_returns_provided(self):
        """Test CashReturn object with provided cash returns."""
        cash_model = CashReturn(self.returns.iloc[:, -1])
        cvxpy_expression = self.boilerplate(cash_model)
        self.w_plus.value = np.random.randn(self.N)
        cash_model.values_in_time_recursive(
            t=self.returns.index[123], past_returns=None)
        cr = self.returns.iloc[123, -1]
        self.assertTrue(cvxpy_expression.value == cr * (
            self.w_plus[-1].value))

    def test_returns_forecast(self):
        """Test ReturnsForecast object with provided assets' returns."""
        alpha_model = ReturnsForecast(self.returns.iloc[:, :-1])
        cvxpy_expression = self.boilerplate(alpha_model)
        alpha_model.values_in_time_recursive(
            t=self.returns.index[123], past_returns=None)
        self.w_plus.value = np.random.randn(self.N)
        print(cvxpy_expression.value)
        print(self.w_plus[:-1].value @ self.returns.iloc[123][:-1])
        self.assertTrue(np.isclose(cvxpy_expression.value,
                        self.w_plus[:-1].value @ self.returns.iloc[123][:-1]))

    def test_full_returns_forecast(self):
        """Test ReturnsForecast object with historical mean forecasts."""
        alpha_model = ReturnsForecast()
        cvxpy_expression = self.boilerplate(alpha_model)
        t = self.returns.index[123]
        alpha_model.values_in_time_recursive(
            t=t, past_returns=self.returns.loc[self.returns.index < t])
        self.w_plus.value = np.random.uniform(size=self.N)
        self.w_plus.value /= sum(self.w_plus.value)
        myforecast = self.returns.iloc[:, :-
                                       1].loc[self.returns.index < t].mean()
        self.assertTrue(np.isclose(cvxpy_expression.value,
                        self.w_plus.value[:-1] @ myforecast))

    def test_returns_forecast_error(self):
        """Test ReturnsForecastError object with provided values."""
        delta = self.returns.iloc[:, :-
                                  1].std(ddof=0) / np.sqrt(len(self.returns))
        error_risk = ReturnsForecastError(delta)
        cvxpy_expression = self.boilerplate(error_risk)
        error_risk.values_in_time_recursive(t='ciao', past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        self.assertTrue(np.isclose(cvxpy_expression.value, np.abs(
            self.w_plus_minus_w_bm.value[:-1]) @ delta))

    def test_full_returns_forecast_error(self):
        """Test ReturnsForecastError object with as forecast the std of the.

        mean estimator.
        """
        error_risk = ReturnsForecastError()
        cvxpy_expression = self.boilerplate(error_risk)
        t = self.returns.index[123]
        past_returns = self.returns.loc[self.returns.index < t]
        error_risk.values_in_time_recursive(t=t, past_returns=past_returns)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        delta = past_returns.std(ddof=0) / np.sqrt(past_returns.count())
        print(cvxpy_expression.value)
        print(np.abs(self.w_plus_minus_w_bm.value[:-1]) @ delta[:-1])
        self.assertTrue(np.isclose(cvxpy_expression.value, np.abs(
            self.w_plus_minus_w_bm.value[:-1]) @ delta[:-1]))


if __name__ == '__main__':

    unittest.main() # pragma: no cover
