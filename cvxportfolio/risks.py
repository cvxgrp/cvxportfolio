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

from .estimator import DataEstimator
import logging
import warnings

import scipy.linalg

import cvxpy as cp
import numpy as np
import pandas as pd

from .costs import BaseCost
from .forecast import HistoricalVariance, HistoricalFactorizedCovariance

logger = logging.getLogger(__name__)


__all__ = [
    "FullCovariance",
    "DiagonalCovariance",
    "FactorModelCovariance",
    "RiskForecastError",
    "WorstCaseRisk",
]


class BaseRiskModel(BaseCost):
    pass


class FullCovariance(BaseRiskModel):
    """Quadratic risk model with full covariance matrix.

    :param Sigma: DataFrame of covariance matrices
        supplied by the user, or None if fitting from the past data.
        The DataFrame can either represents a single constant covariance matrix
        or one for each point in time.
    :type Sigma: pandas.DataFrame or None


    """
    # r"""Quadratic risk model with full covariance matrix.
    #
    # This class represents the term :math:`\Sigma_t`, *i.e.,*
    # the :math:`(n-1) \times (n-1)` positive semi-definite matrix
    # which estimates the covariance of the (non-cash) assets' returns.
    # :ref:`Optimization-based policies` use this, as is explained
    # in Chapter 4 and 5 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    #
    # The user can either supply a :class:`pandas.DataFrame` with the covariance matrix
    # (constant or varying in time) computed externally (for example
    # with some machine learning technique) or let this class estimate the covariance from the data.
    # The latter is the default behavior.
    #
    # This class implements three ways to compute the covariance matrix from the past returns. The
    # computation is repeated at each point in time :math:`t` of a :class:`BackTest` using only
    # the past returns available at that point: :math:`r_{t-1}, r_{t-2}, \ldots`.
    #
    # * *rolling covariance*, using :class:`pandas.DataFrame.rolling.cov`. This is done
    #   if the user specifies the ``rolling`` argument.
    # * *exponential moving window covariance*, using :class:`pandas.DataFrame.ewm.cov`. This is done
    #   if the user specifies the ``halflife`` argument (``rolling`` takes precedence).
    # * *full historical covariance*, using :class:`pandas.DataFrame.cov`. This is the default
    #   behavior if no arguments are specified.
    #
    # If there are missing data in the historical returns the estimated covariance may not
    # be positive semi-definite. We correct it by projecting on the positive semi-definite
    # cone (*i.e.*, we set the negative eigenvalues of the resulting :math:`\Sigma_t` to zero).
    #
    # :param Sigma: :class:`pandas.DataFrame` of covariance matrices
    #     supplied by the user. The DataFrame either represents a single (constant) covariance matrix
    #     or one for each point in time. In the latter case the DataFrame must have a :class:`pandas.MultiIndex`
    #     where the first level is a :class:`pandas.DatetimeIndex`. If ``None`` (the default)
    #     the covariance matrix is computed from past returns.
    # :type Sigma: pandas.DataFrame or None
    # :param rolling: if it is not ``None`` the covariance matrix will be estimated
    #     on a rolling window of size ``rolling`` of the past returns.
    # :type rolling: int or None
    # :param halflife: if it is not ``None`` the covariance matrix will be estimated
    #     on an exponential moving window of the past returns with half-life ``halflife``.
    #     If ``rolling`` is specified it takes precedence over ``halflife``. If both are ``None`` the full history
    #     will be used for estimation.
    # :type halflife: int or None
    # :param kappa: the multiplier for the associated forecast error risk
    #     (see pages 32-33 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_).
    #     If ``float`` a passed it is treated as a constant, if ``pandas.Series`` with ``pandas.DateTime`` index
    #     it varies in time, if ``None`` the forecast error risk term will not be compiled.
    # :type kappa: float or pandas.Series or None
    # :param kelly: correct the covariance matrix with the term :math:`\mu\mu^T`, as is explained
    #     in page 28 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_,
    #     to match the second term of the Taylor expansion of the portfolio log-return. Default
    #     is ``False``, corresponding to classical mean-variance optimization. If ``True``, it
    #     estimates :math:`\mu` with the same technique as :math:`\Sigma`, *i.e.*, with rolling window
    #     average, exponential moving window average, or an average of the full history.
    # :type kelly: bool
    # """

    def __init__(self, Sigma=None, kelly=True):

        if not Sigma is None:
            self.Sigma = DataEstimator(Sigma)
            self._alreadyfactorized = False
        else:
            self.Sigma = HistoricalFactorizedCovariance(kelly=kelly)
            self._alreadyfactorized = True

    def _pre_evaluation(self, universe, backtest_times):
        self.Sigma_sqrt = cp.Parameter((len(universe)-1, len(universe)-1))

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""

        if self._alreadyfactorized:
            self.Sigma_sqrt.value = self.Sigma.current_value
        else:
            Sigma = self.Sigma.current_value
            eigval, eigvec = np.linalg.eigh(Sigma)
            eigval = np.maximum(eigval, 0.)
            self.Sigma_sqrt.value = eigvec @ np.diag(np.sqrt(eigval))

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.cvxpy_expression = cp.sum_squares(
            self.Sigma_sqrt.T @ w_plus_minus_w_bm[:-1])
        return self.cvxpy_expression


class RiskForecastError(BaseRiskModel):
    """Risk forecast error. 

    Implements the model defined in page 31 of the book. Takes same arguments
    as :class:`DiagonalCovariance`.

    :param sigma_squares: per-stock variances, indexed by time if DataFrame.
        If None it will be fitted on past data.
    :type sigma_squares: pd.DataFrame or pd.Series or None
    """

    def __init__(self, sigma_squares=None):
        if sigma_squares is None:
            self.sigma_squares = HistoricalVariance(kelly=True)  # None None
        else:
            self.sigma_squares = DataEstimator(sigma_squares)
        # self.standard_deviations = ParameterEstimator(standard_deviations)
        # self.zeroforcash=True
        # self.kelly=True

    def _pre_evaluation(self, universe, backtest_times):
        self.sigmas_parameter = cp.Parameter(
            len(universe)-1, nonneg=True)  # +self.kelly))

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""

        # if self.sigma_squares is None:
        #     sigma_squares = past_returns.var(ddof=0)
        #     if self.kelly:
        #         mean = past_returns.mean()
        #         sigma_squares += mean**2
        #     if self.zeroforcash:
        #         sigma_squares.iloc[-1] = 0.
        #     sigma_squares = sigma_squares.values
        # else:
        #     sigma_squares = self.sigma_squares.current_value

        sigma_squares = self.sigma_squares.current_value

        self.sigmas_parameter.value = np.sqrt(sigma_squares)

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        return cp.square(cp.abs(w_plus_minus_w_bm[:-1]).T @ self.sigmas_parameter)


class DiagonalCovariance(BaseRiskModel):
    """Diagonal covariance matrix, user-provided or fit from data.

    :param sigma_squares: per-stock variances, indexed by time if DataFrame.
        If None it will be fitted on past data.
    :type sigma_squares: pd.DataFrame or pd.Series or None 
    """

    def __init__(self, sigma_squares=None):
        if not sigma_squares is None:
            self.sigma_squares = DataEstimator(sigma_squares)
        else:
            self.sigma_squares = HistoricalVariance(kelly=True)  # None
        # self.zeroforcash = True
        # self.kelly = True
        # self.standard_deviations = ParameterEstimator(standard_deviations)

    def _pre_evaluation(self, universe, backtest_times):
        self.sigmas_parameter = cp.Parameter(len(universe)-1)  # +self.kelly))

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""
        # super()._recursive_values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)

        # if self.sigma_squares is None:
        #     sigma_squares = past_returns.var(ddof=0)
        #     if self.kelly:
        #         mean = past_returns.mean()
        #         sigma_squares += mean**2
        #     if self.zeroforcash:
        #         sigma_squares[-1] = 0.
        #     sigma_squares = sigma_squares.values
        # else:
        #     sigma_squares = self.sigma_squares.current_value

        sigma_squares = self.sigma_squares.current_value

        self.sigmas_parameter.value = np.sqrt(sigma_squares)

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        return cp.sum_squares(cp.multiply(w_plus_minus_w_bm[:-1], self.sigmas_parameter))


class FactorModelCovariance(BaseRiskModel):
    """Factor model covariance, either user-provided or fitted from the data.

    It has the structure

    :math:`F \Sigma_{F} F^T + \mathbf{diag}(d)`

    where the factors exposure :math:`F` has as many rows as the number of assets and as many
    columns as the number of factors,
    the factors covariance matrix :math:`Sigma_{F}` is positive semi-definite,
    and the idyosyncratic variances vector :math:`d` is non-negative. 

    The advantage of this risk model over the standard :class:`FullCovariance` is mostly
    computational. When well-specified (as we do here) it costs much less to solve an 
    optimization problem with this model than with a full covariance. It is a standard 
    model that has been used for many decades in the portfolio optimization community
    and multiple vendors exist for covariance matrices in this form. We also provide
    the functionality to compute this automatically (which happens if you only specify
    the number of factors to the constructor) with a standard PCA of the historical 
    covariance at each point in time of the backtest (only looking at past returns).

    :param F: factors exposure matrix either constant or varying in time. If constant
        use a dataframe where the index are the factors and the columns are the asset names.
        If varying in time use a pandas multiindexed dataframe where the first index level
        is time and the second level are the factors. The columns
        should always be the asset names. If None the constructor will default to fit the model from
        past returns at each point of the backtest.
    :type F: pd.DataFrame or None
    :param Sigma_F: factors covariance matrix either constant or varying in time. If varying in time
        use a multiindexed dataframe where the first index level is time, and the second are the
        factors (like the columns). If None it is assumed that :math:`Sigma_F` is the identity matrix: Leaving
        this to None will not trigger automatic fit of the model. You can also have a factors
        exposure matrix that is fixed in time and a factors covariance that instead changes in time,
        or the opposite.
    :type Sigma_F: pd.DataFrame or None
    :param d: idyosyncratic variances either constant or varying in time. If constant use a pandas series,
        if varying in time use a pandas dataframe where the index is time and the columns are the asset
        names. You can have this varying in time and the exposures or the factors covariance fixed, or the
        opposite. If you leave this to None you will trigger automatic fit of the model. If you wish
        to have no idyosyncratic variances you can for example just pass 0.
    :type d: pd.Series or pd.DataFrame or None
    :param num_factors: number of factors (columns of F) that are obtained when fitting the model automatically
    :type num_factors: int    
    :param kelly: when fitting the model you can perform a PCA on the covariance defined in the standard way
        with this equal to False, or use the the quadratic term of the Taylor approximation of the Kelly
        gambling objective with this equal to True. The difference is very small and it's better both
        computationally and in terms of portfolio performance to leave this to True.
    :type kelly: bool
    """

    def __init__(self, F=None, d=None, Sigma_F=None, num_factors=1, kelly=True):
        self.F = F if F is None else DataEstimator(F, compile_parameter=True)
        self.d = d if d is None else DataEstimator(d)
        self.Sigma_F = Sigma_F if Sigma_F is None else DataEstimator(Sigma_F, ignore_shape_check=True)
        if (self.F is None) or (self.d is None):
            self._fit = True
            self.Sigma = HistoricalFactorizedCovariance(kelly=kelly)
        else:
            self._fit = False
        self.num_factors = num_factors

    def _pre_evaluation(self, universe, backtest_times):
        self.idyosync_sqrt_parameter = cp.Parameter(len(universe)-1)
        effective_num_factors = min(self.num_factors, len(universe)-1)
        if self._fit:
            self.F_parameter = cp.Parameter(
                (effective_num_factors, len(universe)-1))
        else:
            if self.Sigma_F is None:
                self.F_parameter = self.F.parameter
            else:
                # we could refactor the code here so we don't create duplicate parameters
                self.F_parameter = cp.Parameter(self.F.parameter.shape)

    def _values_in_time(self, t, past_returns, **kwargs):
        if self._fit:
            Sigmasqrt = self.Sigma.current_value
            # numpy eigendecomposition has largest eigenvalues last
            self.F_parameter.value = Sigmasqrt[:, -self.num_factors:].T
            d = np.sum(Sigmasqrt[:, :-self.num_factors]**2, axis=1)
        else:
            d = self.d.current_value
            if not (self.Sigma_F is None):
                self.F_parameter.value = (
                    self.F.parameter.value.T @ np.linalg.cholesky(self.Sigma_F.current_value)).T

        self.idyosync_sqrt_parameter.value = np.sqrt(d)

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.expression = cp.sum_squares(cp.multiply(
            self.idyosync_sqrt_parameter, w_plus_minus_w_bm[:-1]))
        assert self.expression.is_dcp(dpp=True)

        self.expression += cp.sum_squares(self.F_parameter @
                                          w_plus_minus_w_bm[:-1])
        assert self.expression.is_dcp(dpp=True)

        return self.expression


class WorstCaseRisk(BaseRiskModel):
    """Select the most restrictive risk model for each value of the allocation vector.

    Given a list of risk models, penalize the portfolio allocation by the
    one with highest risk value at the solution point. If uncertain about
    which risk model to use this procedure can be an easy solution.

    :param riskmodels: risk model instances on which to compute the worst-case
        risk.
    :type riskmodels: list 
    """

    def __init__(self, riskmodels):
        self.riskmodels = riskmodels

    def _recursive_pre_evaluation(self, universe, backtest_times):
        """Initialize objects."""
        for risk in self.riskmodels:
            risk._recursive_pre_evaluation(universe, backtest_times)

    def _recursive_values_in_time(self, **kwargs):
        """Update parameters."""
        for risk in self.riskmodels:
            risk._recursive_values_in_time(**kwargs)

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        risks = [risk._compile_to_cvxpy(w_plus, z, w_plus_minus_w_bm)
                 for risk in self.riskmodels]
        return cp.max(cp.hstack(risks))
