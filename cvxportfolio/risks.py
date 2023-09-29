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

import logging

import cvxpy as cp
import numpy as np

from .costs import Cost
from .estimator import DataEstimator
from .forecast import (HistoricalFactorizedCovariance,
                       HistoricalLowRankCovarianceSVD, HistoricalVariance,
                       project_on_psd_cone_and_factorize)

logger = logging.getLogger(__name__)


__all__ = [
    "FullCovariance",
    "DiagonalCovariance",
    "FactorModelCovariance",
    "RiskForecastError",
    "WorstCaseRisk",
]


class BaseRiskModel(Cost):
    pass


class FullCovariance(BaseRiskModel):
    r"""Quadratic risk model with full covariance matrix.

    It represents the objective term:

    .. math::
        {(w^+_t - w^\text{b}_t )}^T \Sigma_t (w^+_t - w^\text{b}_t)

    where :math:`w^+_t` and :math:`w^\text{b}_t` are the post-trade
    and the benchmark weights, respectively, at time :math:`t`.

    :param Sigma: DataFrame of covariance matrices
        supplied by the user, or by default Covariance fitted from the past data.
        The DataFrame can either represents a single constant covariance matrix
        or one for each point in time. If it is a class we instantiate
        it with default parameters. At each time :math:`t` we project the value of
        :math:`\Sigma_t` on the cone of positive semi-definite matrices.
    :type Sigma: pandas.DataFrame or Estimator
    """

    def __init__(self, Sigma=HistoricalFactorizedCovariance):

        if isinstance(Sigma, type):
            Sigma = Sigma()

        self._alreadyfactorized = hasattr(Sigma, 'FACTORIZED')\
            and Sigma.FACTORIZED

        self.Sigma = DataEstimator(Sigma)

    def initialize_estimator(self, universe, trading_calendar):
        self.Sigma_sqrt = cp.Parameter((len(universe)-1, len(universe)-1))

    def values_in_time(self, t, past_returns, **kwargs):

        if self._alreadyfactorized:
            self.Sigma_sqrt.value = self.Sigma.current_value
        else:
            self.Sigma_sqrt.value = project_on_psd_cone_and_factorize(
                self.Sigma.current_value)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.cvxpy_expression = cp.sum_squares(
            self.Sigma_sqrt.T @ w_plus_minus_w_bm[:-1])
        return self.cvxpy_expression


class RiskForecastError(BaseRiskModel):
    """Risk forecast error.

    Implements the model defined in page 31 of the book. Takes same arguments
    as :class:`DiagonalCovariance`.

    :param sigma_squares: per-stock variances, indexed by time if DataFrame.
        Default is to use historical variances, using
        past returns at each point in time of a backtest.
    :type sigma_squares: pd.DataFrame or pd.Series or None
    """

    def __init__(self, sigma_squares=HistoricalVariance):

        if isinstance(sigma_squares, type):
            sigma_squares = sigma_squares()

        self.sigma_squares = DataEstimator(sigma_squares)

    def initialize_estimator(self, universe, trading_calendar):
        self.sigmas_parameter = cp.Parameter(
            len(universe)-1, nonneg=True)  # +self.kelly))

    def values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""

        sigma_squares = self.sigma_squares.current_value

        self.sigmas_parameter.value = np.sqrt(sigma_squares)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        return cp.square(cp.abs(w_plus_minus_w_bm[:-1]).T @ self.sigmas_parameter)


class DiagonalCovariance(BaseRiskModel):
    """Diagonal covariance matrix, user-provided or fit from data.

    :param sigma_squares: per-stock variances, indexed by time if
        DataFrame. Default is to use historical variances, using past
        returns at each point in time of a backtest.
    :type sigma_squares: pd.DataFrame or pd.Series or None
    """

    def __init__(self, sigma_squares=HistoricalVariance):

        if isinstance(sigma_squares, type):
            sigma_squares = sigma_squares()
        self.sigma_squares = DataEstimator(sigma_squares)

    def initialize_estimator(self, universe, trading_calendar):
        self.sigmas_parameter = cp.Parameter(len(universe)-1)  # +self.kelly))

    def values_in_time(self, t, past_returns, **kwargs):
        sigma_squares = self.sigma_squares.current_value
        self.sigmas_parameter.value = np.sqrt(sigma_squares)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        return cp.sum_squares(cp.multiply(w_plus_minus_w_bm[:-1],
            self.sigmas_parameter))


class FactorModelCovariance(BaseRiskModel):
    r"""Factor model covariance, either user-provided or fitted from the data.

    It represents the objective term:

    .. math::

        {(w^+_t - w^\text{b}_t )}^T (F \Sigma_{F} F^T + \mathbf{diag}(d)) (w^+_t - w^\text{b}_t)

    where the factors exposure :math:`F` has as many rows as the number of assets and as many
    columns as the number of factors,
    the factors covariance matrix :math:`\Sigma_{F}` is positive semi-definite,
    and the idyosyncratic variances vector :math:`d` is non-negative.

    The advantage of this risk model over the standard :class:`FullCovariance` is mostly
    computational. When well-specified (as we do here) it costs much less to solve an
    optimization problem with this model than with a full covariance. It is a standard
    model that has been used for many decades in the portfolio optimization community
    and multiple vendors exist for covariance matrices in this form. We also provide
    the functionality to compute this automatically (which happens if you only specify
    the number of factors to the constructor) with a standard PCA of the historical
    covariance at each point in time of the backtest (only looking at past returns).

    :param F: Factors exposure matrix either constant or varying in time. If constant
        use a dataframe where the index are the factors and the columns are the asset names.
        If varying in time use a pandas multiindexed dataframe where the first index level
        is time and the second level are the factors. The columns
        should always be the asset names. If None the constructor will default to fit the model from
        past returns at each point of the backtest.
    :type F: pandas.DataFrame or None
    :param Sigma_F: Factors covariance matrix either constant or varying in time. If varying in time
        use a multiindexed dataframe where the first index level is time, and the second are the
        factors (like the columns). If None it is assumed that :math:`Sigma_F` is the identity matrix: Leaving
        this to None will not trigger automatic fit of the model. You can also have a factors
        exposure matrix that is fixed in time and a factors covariance that instead changes in time,
        or the opposite.
    :type Sigma_F: pandas.DataFrame or None
    :param d: Idyosyncratic variances either constant or varying in time. If constant use a pandas series,
        if varying in time use a pandas dataframe where the index is time and the columns are the asset
        names. You can have this varying in time and the exposures or the factors covariance fixed, or the
        opposite. If you leave this to None you will trigger automatic fit of the model. If you wish
        to have no idyosyncratic variances you can for example just pass 0.
    :type d: pandas.Series or pandas.DataFrame or None
    :param num_factors: Number of factors (columns of F) that are obtained when
        fitting the model automatically (otherwise it is ignored).
    :type num_factors: int
    :param Sigma: Only relevant if F or d are None. Same as the parameter
        passed to :class:`FullCovariance` (by default,
        historical covariance fitted at each point in time). We take its PCA for the low-rank
        model, and the remaining factors are used to estimate the diagonal, as
        is explained at pages 59-60 of the book. If it is a class, we instantiate
        it with default parameters.
    :type Sigma: pandas.DataFrame or Estimator
    :param F_and_d_Forecaster: Only relevant if F or d are None, and Sigma is None. Forecaster
        that at each point in time produces estimate of F and d. By default we use a SVD-based
        forecaster that is equivalent to :class:`HistoricalFactorizedCovariance` if there
        are no missing values. If you pass a class, it will be instantiated with ``num_factors``.
    :type F_and_d_Forecaster: Estimator
    """

    def __init__(self, F=None, d=None, Sigma_F=None, num_factors=1,
            Sigma=HistoricalFactorizedCovariance, F_and_d_Forecaster=HistoricalLowRankCovarianceSVD):
        self.F = F if F is None else DataEstimator(F, compile_parameter=True)
        self.d = d if d is None else DataEstimator(d)
        self.Sigma_F = Sigma_F if Sigma_F is None else DataEstimator(Sigma_F, ignore_shape_check=True)
        if (self.F is None) or (self.d is None):
            self._fit = True
            if Sigma is None:
                if isinstance(F_and_d_Forecaster, type):
                    F_and_d_Forecaster = F_and_d_Forecaster(num_factors=num_factors)
                self.F_and_d_Forecaster = F_and_d_Forecaster
            else:
                if isinstance(Sigma, type):
                    Sigma = Sigma()
                self._alreadyfactorized = hasattr(Sigma, 'FACTORIZED')\
                    and Sigma.FACTORIZED
                self.Sigma = DataEstimator(Sigma)
            self.num_factors = num_factors
        else:
            self._fit = False

    def initialize_estimator(self, universe, trading_calendar):
        self.idyosync_sqrt_parameter = cp.Parameter(len(universe)-1)
        if self._fit:
            effective_num_factors = min(self.num_factors, len(universe)-1)
            self.F_parameter = cp.Parameter(
                (effective_num_factors, len(universe)-1))
        else:
            if self.Sigma_F is None:
                self.F_parameter = self.F.parameter
            else:
                # we could refactor the code here so we don't create duplicate parameters
                self.F_parameter = cp.Parameter(self.F.parameter.shape)

    def values_in_time(self, t, past_returns, **kwargs):

        if self._fit:
            if hasattr(self, 'F_and_d_Forecaster'):
                self.F_parameter.value, d = self.F_and_d_Forecaster.current_value
            else:
                Sigmasqrt = self.Sigma.current_value if self._alreadyfactorized\
                    else project_on_psd_cone_and_factorize(
                        self.Sigma.current_value)
                # numpy eigendecomposition has largest eigenvalues last
                self.F_parameter.value = Sigmasqrt[:, -self.num_factors:].T
                d = np.sum(Sigmasqrt[:, :-self.num_factors]**2, axis=1)
        else:
            d = self.d.current_value
            if not (self.Sigma_F is None):
                self.F_parameter.value = (
                    self.F.parameter.value.T @ np.linalg.cholesky(
                        self.Sigma_F.current_value)).T

        self.idyosync_sqrt_parameter.value = np.sqrt(d)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.expression = cp.sum_squares(cp.multiply(
            self.idyosync_sqrt_parameter, w_plus_minus_w_bm[:-1]))
        assert self.expression.is_dcp(dpp=True)

        self.expression += cp.sum_squares(self.F_parameter @
                                          w_plus_minus_w_bm[:-1])
        assert self.expression.is_dcp(dpp=True)

        return self.expression


class WorstCaseRisk(BaseRiskModel):
    """Select the most restrictive risk model for each value of the allocation.

    vector.

    Given a list of risk models, penalize the portfolio allocation by
    the one with highest risk value at the solution point. If uncertain
    about which risk model to use this procedure can be an easy
    solution.

    :param riskmodels: risk model instances on which to compute the
        worst-case risk.
    :type riskmodels: list
    """

    def __init__(self, riskmodels):
        self.riskmodels = riskmodels

    def initialize_estimator_recursive(self, universe, trading_calendar):
        """Initialize objects."""
        for risk in self.riskmodels:
            risk.initialize_estimator_recursive(universe, trading_calendar)

    def values_in_time_recursive(self, **kwargs):
        """Update parameters."""
        for risk in self.riskmodels:
            risk.values_in_time_recursive(**kwargs)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        risks = [risk.compile_to_cvxpy(w_plus, z, w_plus_minus_w_bm)
                 for risk in self.riskmodels]
        return cp.max(cp.hstack(risks))
