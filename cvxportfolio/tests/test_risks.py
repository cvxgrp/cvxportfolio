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

"""Unit tests for the risk objects."""

import unittest
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from cvxportfolio.risks import *

USED_RETURNS = 10


class TestRisks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        cls.returns = pd.read_csv(Path(
            __file__).parent / "returns.csv", index_col=0, parse_dates=[0]).iloc[:, :USED_RETURNS]
        cls.w_plus = cp.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
        cls.z = cp.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]

    def boilerplate(self, model):
        model._recursive_pre_evaluation(
            universe=self.returns.columns, backtest_times=self.returns.index)
        return model._compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)

    def test_full_sigma(self):
        historical_covariances = self.returns.iloc[:,
                                                   :-1].rolling(50).cov(ddof=0).dropna()
        risk_model = FullCovariance(historical_covariances)

        cvxpy_expression = self.boilerplate(risk_model)

        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_covariances.index[123][0]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model._recursive_values_in_time(t=t, past_returns='Hello!')

        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value[:-1] @
                                   historical_covariances.loc[t] @ self.w_plus_minus_w_bm.value[:-1]))

    def test_full_estimated_sigma(self):

        risk_model = FullCovariance()

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        # N = returns.shape[1]
        # returns.iloc[:, -1] = 0.

        # w_plus = cp.Variable(N)
        # risk_model._recursive_pre_evaluation(
        #     returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
        # cvxpy_expression = risk_model._compile_to_cvxpy(w_plus, None, None)

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = past.iloc[:, :-1].T @ past.iloc[:, :-1] / len(past)

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model._recursive_values_in_time(t=t, past_returns=past)
        print(cvxpy_expression.value)

        print(self.w_plus_minus_w_bm.value[:-1] @
              should_be @ self.w_plus_minus_w_bm.value[:-1])

        self.assertTrue(np.isclose(cvxpy_expression.value,
                                   self.w_plus_minus_w_bm.value[:-1] @ should_be @ self.w_plus_minus_w_bm.value[:-1]))

    def test_diagonal_covariance(self):

        # N = returns.shape[1]
        # returns.iloc[:, -1] = 0.

        historical_variances = self.returns.iloc[:,
                                                 :-1].rolling(50).var().shift(1).dropna()
        risk_model = DiagonalCovariance(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model._recursive_values_in_time(t=t, past_returns='hello')

        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value[:-1] @
                                   np.diag(historical_variances.loc[t]) @ self.w_plus_minus_w_bm.value[:-1]))

    def test_full_diagonal_covariance(self):

        risk_model = DiagonalCovariance()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.

        risk_model._recursive_values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value @
                                   np.diag(should_be) @ self.w_plus_minus_w_bm.value))

    def test_forecast_error(self):

        historical_variances = (
            self.returns.iloc[:, :-1]**2).rolling(50).mean().shift(1).dropna()

        risk_model = RiskForecastError(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)

        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model._recursive_values_in_time(t=t, past_returns='hello')

        print(cvxpy_expression.value)
        print((np.abs(
            self.w_plus_minus_w_bm.value[:-1]) @ np.sqrt(historical_variances.loc[t]))**2)
        self.assertTrue(np.isclose(cvxpy_expression.value,

                                   (np.abs(
                                       self.w_plus_minus_w_bm.value[:-1]) @ np.sqrt(historical_variances.loc[t]))**2

                                   ))

    def test_full_forecast_error(self):

        risk_model = RiskForecastError()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')

        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.

        risk_model._recursive_values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value,

                                   (np.abs(self.w_plus_minus_w_bm.value)
                                    @ np.sqrt(should_be))**2

                                   ))

    def test_low_rank_covariance(self):

        F = pd.DataFrame(np.random.randn(2, self.N-1),
                         columns=self.returns.columns[:-1])
        d = pd.Series(np.random.uniform(size=(self.N-1)),
                      self.returns.columns[:-1])
        risk_model = FactorModelCovariance(F=F, d=d)

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        risk_model._recursive_values_in_time(
            t=self.returns.index[12], past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value,
                                   self.w_plus_minus_w_bm.value[:-1] @ np.diag(d) @ self.w_plus_minus_w_bm.value[:-1] +
                                   ((F @ self.w_plus_minus_w_bm.value[:-1])**2).sum()))

    def test_low_rank_covariance_with_SigmaF(self):

        F = pd.DataFrame(np.random.randn(2, self.N-1),
                         columns=self.returns.columns[:-1])
        d = pd.Series(np.random.uniform(size=(self.N-1)),
                      self.returns.columns[:-1])
        SigmaF = np.random.randn(2, 2)
        SigmaF = SigmaF.T @ SigmaF
        risk_model = FactorModelCovariance(F=F, d=d, Sigma_F=SigmaF)

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        risk_model._recursive_values_in_time(
            t=self.returns.index[12], past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value,
                                   self.w_plus_minus_w_bm.value[:-1] @ np.diag(d) @ self.w_plus_minus_w_bm.value[:-1] +
                                   self.w_plus_minus_w_bm.value[:-1].T @ F.T @ SigmaF @ F @ self.w_plus_minus_w_bm.value[:-1])
                        )

    def test_estimated_low_rank_covariance(self):

        risk_model = FactorModelCovariance()  # normalize=False)

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')

        past = self.returns.loc[self.returns.index < t]

        FS = past.T @ past / len(past)
        FS.iloc[:, -1] = 0.
        FS.iloc[-1, :] = 0.
        eigval, eigvec = np.linalg.eigh(FS)
        F = np.sqrt(eigval[-1]) * eigvec[:, -1]
        d = np.diag(FS) - F**2

        risk_model._recursive_values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value,
                                   self.w_plus_minus_w_bm.value @ np.diag(d) @ self.w_plus_minus_w_bm.value +
                                   ((F @ self.w_plus_minus_w_bm.value)**2).sum()))

    def test_worst_case_risk(self):

        risk_model0 = FullCovariance()
        risk_model1 = DiagonalCovariance()
        worst_case = WorstCaseRisk([risk_model0, risk_model1])

        cvxpy_expression = self.boilerplate(worst_case)

        assert cvxpy_expression.is_convex()

        cvxpy_expression0 = risk_model0._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        cvxpy_expression1 = risk_model1._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)

        self.w_plus_minus_w_bm.value = np.ones(self.N)

        t = pd.Timestamp('2014-06-02')

        worst_case._recursive_values_in_time(
            t=t, past_returns=self.returns.loc[self.returns.index < t])

        print(cvxpy_expression.value)
        print(cvxpy_expression0.value)
        print(cvxpy_expression1.value)
        assert (cvxpy_expression.value == cvxpy_expression0.value)
        assert (cvxpy_expression.value > cvxpy_expression1.value)


if __name__ == '__main__':
    unittest.main()
