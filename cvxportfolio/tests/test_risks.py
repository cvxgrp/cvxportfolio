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


VALUES_IN_TIME_DUMMY_KWARGS = {
    'current_weights': None,
    'past_returns': None,
    'current_prices': None,
    'past_volumes': None,
    'current_portfolio_value': 1000,
}


class TestSOCConstraints(CvxportfolioTest):
    """Tests for SOC compilation of CostInequalityConstraint."""

    def _build_risk_constraint(self, constraint, t):
        """Initialize a risk inequality constraint and return the cvxpy object.

        :param constraint: A :class:`CostInequalityConstraint` instance.
        :param t: Timestamp used to evaluate parameters.
        :returns: The compiled cvxpy constraint.
        """
        constraint.initialize_estimator_recursive(
            universe=self.returns.columns,
            trading_calendar=self.returns.index)
        cvxpy_constr = constraint.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)
        constraint.values_in_time_recursive(
            t=t, **VALUES_IN_TIME_DUMMY_KWARGS)
        return cvxpy_constr

    @staticmethod
    def _has_norm2(expr_or_constr):
        """Return True if any atom in the expression tree is a 2-norm type.

        This handles both the stable ``norm2`` atom name and the
        ``PnormApprox`` class used in development CVXPY builds.
        """
        _NORM_NAMES = frozenset({
            'norm2', 'Norm2', 'pnorm', 'Pnorm', 'PnormApprox',
        })

        def _walk(expr):
            if type(expr).__name__ in _NORM_NAMES:
                return True
            if hasattr(expr, 'args'):
                return any(_walk(a) for a in expr.args)
            return False
        # Constraints expose args directly (lhs, rhs)
        if hasattr(expr_or_constr, 'args'):
            return any(_walk(a) for a in expr_or_constr.args)
        return False  # pragma: no cover

    def test_full_covariance_soc_compilation(self):
        """FullCovariance <= limit uses SOC (norm2) form, not sum_squares."""
        historical_covariances = self.returns.iloc[
            :, :-1].rolling(50).cov(ddof=0).dropna()
        risk_model = cvx.FullCovariance(historical_covariances)
        limit = 0.05
        constr = risk_model <= limit

        t = historical_covariances.index[123][0]
        cvxpy_constr = self._build_risk_constraint(constr, t)

        # Must be DCP and DPP
        self.assertTrue(cvxpy_constr.is_dcp())
        self.assertTrue(cvxpy_constr.is_dcp(dpp=True))

        # LHS should involve a norm2 atom, not fall back to scalar
        self.assertTrue(self._has_norm2(cvxpy_constr))

        # Numeric equivalence: norm2(sigma_sqrt^T @ w)^2 == w^T Sigma w
        self.w_plus_minus_w_bm.value = np.random.default_rng(0).standard_normal(
            self.N)
        w = self.w_plus_minus_w_bm.value[:-1]
        sigma = historical_covariances.loc[t].values
        expected_lhs_sq = float(w @ sigma @ w)

        # args[0] of the norm2 constraint is the norm2 expression
        lhs_val = float(cvxpy_constr.args[0].value)
        self.assertAlmostEqual(lhs_val ** 2, expected_lhs_sq, places=10)

        # sqrt_value_param must equal sqrt(limit)
        self.assertAlmostEqual(
            float(cvxpy_constr.args[1].value), np.sqrt(limit), places=12)

    def test_diagonal_covariance_soc_compilation(self):
        """DiagonalCovariance <= limit uses the SOC form."""
        historical_variances = self.returns.iloc[
            :, :-1].rolling(50).var().shift(1).dropna()
        risk_model = cvx.DiagonalCovariance(historical_variances)
        limit = 0.03
        constr = risk_model <= limit

        t = historical_variances.index[123]
        cvxpy_constr = self._build_risk_constraint(constr, t)

        self.assertTrue(cvxpy_constr.is_dcp())
        self.assertTrue(cvxpy_constr.is_dcp(dpp=True))
        self.assertTrue(self._has_norm2(cvxpy_constr))

        self.w_plus_minus_w_bm.value = np.random.default_rng(1).standard_normal(
            self.N)
        w = self.w_plus_minus_w_bm.value[:-1]
        variances = historical_variances.loc[t].values
        expected_lhs_sq = float((w ** 2) @ variances)

        lhs_val = float(cvxpy_constr.args[0].value)
        self.assertAlmostEqual(lhs_val ** 2, expected_lhs_sq, places=10)

    def test_factor_model_covariance_soc_compilation(self):
        """FactorModelCovariance <= limit uses cp.hstack + norm2 SOC form."""
        F = pd.DataFrame(
            np.random.default_rng(2).standard_normal((2, self.N - 1)),
            columns=self.returns.columns[:-1])
        d = pd.Series(
            np.random.default_rng(3).uniform(0.001, 0.05, size=self.N - 1),
            index=self.returns.columns[:-1])
        risk_model = cvx.FactorModelCovariance(F=F, d=d)
        limit = 0.04
        constr = risk_model <= limit

        t = self.returns.index[50]
        cvxpy_constr = self._build_risk_constraint(constr, t)

        self.assertTrue(cvxpy_constr.is_dcp())
        self.assertTrue(cvxpy_constr.is_dcp(dpp=True))
        self.assertTrue(self._has_norm2(cvxpy_constr))

        self.w_plus_minus_w_bm.value = np.random.default_rng(4).standard_normal(
            self.N)
        w = self.w_plus_minus_w_bm.value[:-1]
        # risk == ||F w||^2 + ||diag(sqrt(d)) w||^2
        expected_lhs_sq = (
            float(np.sum((F.values @ w) ** 2)) +
            float(np.sum(d.values * w ** 2)))

        lhs_val = float(cvxpy_constr.args[0].value)
        self.assertAlmostEqual(lhs_val ** 2, expected_lhs_sq, places=10)

    def test_soc_fallback_for_custom_cost(self):
        """Custom Cost without _soc_expression falls back to scalar path."""
        import cvxpy as cp
        from cvxportfolio.costs import Cost

        class _SquaredCashWeight(Cost):
            """Trivial cost: square of cash weight (no SOC override)."""
            def compile_to_cvxpy(  # pylint: disable=arguments-differ
                    self, w_plus, **kwargs):
                return cp.square(w_plus[-1])

        custom_cost = _SquaredCashWeight()
        constr = custom_cost <= 0.1

        constr.initialize_estimator_recursive(
            universe=self.returns.columns,
            trading_calendar=self.returns.index)
        cvxpy_constr = constr.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)
        constr.values_in_time_recursive(
            t=self.returns.index[0], **VALUES_IN_TIME_DUMMY_KWARGS)

        self.assertTrue(cvxpy_constr.is_dcp())
        self.assertTrue(cvxpy_constr.is_dcp(dpp=True))

        # Must NOT have norm2 — the scalar fallback should be used
        self.assertFalse(self._has_norm2(cvxpy_constr))

        # Verify the constraint evaluates correctly
        self.w_plus.value = np.ones(self.N) / self.N
        cash_weight = self.w_plus.value[-1]
        # Constraint is w[-1]^2 <= 0.1
        self.assertAlmostEqual(
            float(cvxpy_constr.args[0].value), cash_weight ** 2, places=12)

    def test_annualized_volatility_soc_constraint(self):
        """AnnualizedVolatility limit is correctly handled by the SOC path."""
        risk_model = cvx.FullCovariance()
        annvol = 0.10  # 10% annualized
        constr = risk_model <= cvx.AnnualizedVolatility(annvol)

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]

        constr.initialize_estimator_recursive(
            universe=self.returns.columns,
            trading_calendar=self.returns.index)
        cvxpy_constr = constr.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)
        constr.values_in_time_recursive(
            t=t, past_returns=past,
            current_weights=None, current_portfolio_value=1000,
            current_prices=None, past_volumes=None)

        self.assertTrue(cvxpy_constr.is_dcp())
        self.assertTrue(cvxpy_constr.is_dcp(dpp=True))
        self.assertTrue(self._has_norm2(cvxpy_constr))

        # The per-period variance limit is (annvol^2)/ppy.
        # The sqrt_value_param must equal annvol/sqrt(ppy).
        from cvxportfolio.utils import periods_per_year_from_datetime_index
        ppy = periods_per_year_from_datetime_index(past.index)
        expected_t = annvol / np.sqrt(ppy)
        self.assertAlmostEqual(
            float(cvxpy_constr.args[1].value), expected_t, places=10)

    def test_soc_and_scalar_same_optimal_solution(self):
        """SOC and scalar constraint forms give numerically identical solutions.

        Build and solve a small mean-variance problem twice — once with the
        SOC risk constraint and once with the legacy scalar form — and verify
        the optimal portfolio weights agree to high precision.
        """
        import cvxpy as cp
        from cvxportfolio.costs import Cost

        N_assets = 5
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N_assets, N_assets))
        Sigma = A.T @ A / N_assets + np.eye(N_assets) * 0.01
        Sigma_sqrt = np.linalg.cholesky(Sigma)  # lower-tri; Sigma = L L^T

        mu = rng.standard_normal(N_assets) * 0.01
        limit = 0.25  # must exceed equal-weight portfolio risk (~0.15)

        # --- SOC formulation ---
        w_soc = cp.Variable(N_assets)
        soc_param = cp.Parameter(nonneg=True)
        soc_param.value = np.sqrt(limit)
        prob_soc = cp.Problem(
            cp.Maximize(mu @ w_soc),
            [
                cp.sum(w_soc) == 1,
                cp.norm2(Sigma_sqrt.T @ w_soc) <= soc_param,
            ])
        prob_soc.solve()

        # --- Scalar (sum_squares) formulation ---
        w_scalar = cp.Variable(N_assets)
        scalar_param = cp.Parameter(nonneg=True)
        scalar_param.value = limit
        prob_scalar = cp.Problem(
            cp.Maximize(mu @ w_scalar),
            [
                cp.sum(w_scalar) == 1,
                cp.sum_squares(Sigma_sqrt.T @ w_scalar) <= scalar_param,
            ])
        prob_scalar.solve()

        self.assertTrue(prob_soc.status in ('optimal', 'optimal_inaccurate'))
        self.assertTrue(
            prob_scalar.status in ('optimal', 'optimal_inaccurate'))
        np.testing.assert_allclose(
            w_soc.value, w_scalar.value, atol=1e-5,
            err_msg='SOC and scalar formulations gave different solutions')


if __name__ == '__main__':

    unittest.main() # pragma: no cover
