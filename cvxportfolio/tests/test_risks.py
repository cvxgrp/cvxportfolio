# Copyright (C) 2017-2024 Enzo Busseti
# Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
#
## Earlier versions of this module had the following copyright and licensing
## notice, which is subsumed by the above.
##
### Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###    http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
"""Unit tests for the risk objects."""

import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.tests import CvxportfolioTest

USED_RETURNS = 10


class TestRisks(CvxportfolioTest):
    """Test risk objects."""

    def test_full_sigma(self):
        """Test full covariance model."""
        historical_covariances = self.returns.iloc[
            :, :-1].rolling(50).cov(ddof=0).dropna()
        risk_model = cvx.FullCovariance(historical_covariances)

        cvxpy_expression = self.boilerplate(risk_model)

        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_covariances.index[123][0]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model.values_in_time_recursive(t=t, past_returns='Hello!')

        self.assertTrue(np.isclose(
            cvxpy_expression.value,
            self.w_plus_minus_w_bm.value[:-1] @ historical_covariances.loc[t]
                @ self.w_plus_minus_w_bm.value[:-1]))

    def test_full_estimated_sigma(self):
        """Test full covariance model estimated internally."""
        risk_model = cvx.FullCovariance()

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = past.iloc[:, :-1].T @ past.iloc[:, :-1] / len(past)

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model.values_in_time_recursive(t=t, past_returns=past)
        # print(cvxpy_expression.value)

        # print(self.w_plus_minus_w_bm.value[:-1] @
        #      should_be @ self.w_plus_minus_w_bm.value[:-1])

        self.assertTrue(
            np.isclose(cvxpy_expression.value,
            self.w_plus_minus_w_bm.value[:-1] @ should_be
                @ self.w_plus_minus_w_bm.value[:-1]))

    def test_diagonal_covariance(self):
        """Test diagonal covariance model."""
        historical_variances = self.returns.iloc[
            :, :-1].rolling(50).var().shift(1).dropna()
        risk_model = cvx.DiagonalCovariance(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model.values_in_time_recursive(t=t, past_returns='hello')

        self.assertTrue(
            np.isclose(cvxpy_expression.value,
                self.w_plus_minus_w_bm.value[:-1]
                @ np.diag(historical_variances.loc[t])
                @ self.w_plus_minus_w_bm.value[:-1]))

    def test_full_diagonal_covariance(self):
        """Test diagonal covariance estimated internally."""

        risk_model = cvx.DiagonalCovariance()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.

        risk_model.values_in_time_recursive(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(
            np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value
                @ np.diag(should_be) @ self.w_plus_minus_w_bm.value))

    def test_forecast_error(self):
        """Test RiskForecastError term."""

        historical_variances = (
            self.returns.iloc[:, :-1]**2).rolling(50).mean().shift(1).dropna()

        risk_model = cvx.RiskForecastError(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)

        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model.values_in_time_recursive(t=t, past_returns='hello')

        # print(cvxpy_expression.value)
        # print((np.abs(
        #    self.w_plus_minus_w_bm.value[:-1])
        #        @ np.sqrt(historical_variances.loc[t]))**2)
        self.assertTrue(
            np.isclose(cvxpy_expression.value,
            (np.abs(self.w_plus_minus_w_bm.value[:-1])
                @ np.sqrt(historical_variances.loc[t]))**2))

    def test_full_forecast_error(self):
        """Test RiskForecastError term estimated internally."""

        risk_model = cvx.RiskForecastError()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')

        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.

        risk_model.values_in_time_recursive(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(
            np.isclose(cvxpy_expression.value,
            (np.abs(self.w_plus_minus_w_bm.value) @ np.sqrt(should_be))**2 ))

    def test_low_rank_covariance(self):
        """Test FactorModelCovariance."""

        F = pd.DataFrame(np.random.randn(2, self.N-1),
                         columns=self.returns.columns[:-1])
        d = pd.Series(np.random.uniform(size=self.N-1),
                      self.returns.columns[:-1])
        risk_model = cvx.FactorModelCovariance(F=F, d=d)

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        risk_model.values_in_time_recursive(
            t=self.returns.index[12], past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(
            np.isclose(cvxpy_expression.value,
            self.w_plus_minus_w_bm.value[:-1] @ np.diag(d)
                @ self.w_plus_minus_w_bm.value[:-1]
                + ((F @ self.w_plus_minus_w_bm.value[:-1])**2).sum()))

    def test_low_rank_covariance_with_factors_covariance(self):
        """Test FactorModelCovariance with non-identity factors covariance."""

        F = pd.DataFrame(np.random.randn(2, self.N-1),
                         columns=self.returns.columns[:-1])
        d = pd.Series(np.random.uniform(size=self.N-1),
                      self.returns.columns[:-1])
        _ = np.random.randn(2, 2)
        factors_cov = _.T @ _
        risk_model = cvx.FactorModelCovariance(F=F, d=d, Sigma_F=factors_cov)

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        risk_model.values_in_time_recursive(
            t=self.returns.index[12], past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(
            np.isclose(cvxpy_expression.value,
                self.w_plus_minus_w_bm.value[:-1] @ np.diag(d)
                    @ self.w_plus_minus_w_bm.value[:-1]
                    + self.w_plus_minus_w_bm.value[:-1].T @ F.T
                    @ factors_cov @ F @ self.w_plus_minus_w_bm.value[:-1]))

    def test_estimated_low_rank_covariance(self):
        """Test FactorModelCovariance estimated internally."""

        risk_model = cvx.FactorModelCovariance()

        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')

        past = self.returns.loc[self.returns.index < t]

        fullsigma = past.T @ past / len(past)
        fullsigma.iloc[:, -1] = 0.
        fullsigma.iloc[-1, :] = 0.
        eigval, eigvec = np.linalg.eigh(fullsigma)
        F = np.sqrt(eigval[-1]) * eigvec[:, -1]
        d = np.diag(fullsigma) - F**2

        risk_model.values_in_time_recursive(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(
            np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value
                @ np.diag(d) @ self.w_plus_minus_w_bm.value
                + ((F @ self.w_plus_minus_w_bm.value)**2).sum()))

    def test_worst_case_risk(self):
        """Test the worst-case risk object."""

        risk_model0 = cvx.FullCovariance()
        risk_model1 = cvx.DiagonalCovariance()
        worst_case = cvx.WorstCaseRisk([risk_model0, risk_model1])

        cvxpy_expression = self.boilerplate(worst_case)

        assert cvxpy_expression.is_convex()

        cvxpy_expression0 = risk_model0.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)
        cvxpy_expression1 = risk_model1.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)

        self.w_plus_minus_w_bm.value = np.ones(self.N)

        t = pd.Timestamp('2014-06-02')

        worst_case.values_in_time_recursive(
            t=t, past_returns=self.returns.loc[self.returns.index < t],
            current_weights=None, current_portfolio_value=None,
            past_volumes=None, current_prices=None)

        # print(cvxpy_expression.value)
        # print(cvxpy_expression0.value)
        # print(cvxpy_expression1.value)
        self.assertTrue(cvxpy_expression.value == cvxpy_expression0.value)
        self.assertTrue(cvxpy_expression.value > cvxpy_expression1.value)


if __name__ == '__main__':

    unittest.main() # pragma: no cover
