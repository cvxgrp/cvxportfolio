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
            self.alreadyfactorized = False
        else:
            self.Sigma = HistoricalFactorizedCovariance(kelly=kelly) 
            self.alreadyfactorized = True
            
    def _pre_evaluation(self, universe, backtest_times):
        self.Sigma_sqrt = cp.Parameter((len(universe)-1, len(universe)-1))

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""
        
        if self.alreadyfactorized:
            self.Sigma_sqrt.value = self.Sigma.current_value
        else:
            Sigma = self.Sigma.current_value
            eigval, eigvec = np.linalg.eigh(Sigma)
            eigval = np.maximum(eigval, 0.)
            self.Sigma_sqrt.value = eigvec @ np.diag(np.sqrt(eigval))

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.cvxpy_expression = cp.sum_squares(self.Sigma_sqrt.T @ w_plus_minus_w_bm[:-1])
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
            self.sigma_squares = HistoricalVariance(kelly=True) #None None
        else:
            self.sigma_squares = DataEstimator(sigma_squares)
        # self.standard_deviations = ParameterEstimator(standard_deviations)
        # self.zeroforcash=True
        # self.kelly=True
        
    def _pre_evaluation(self, universe, backtest_times):
        self.sigmas_parameter = cp.Parameter(len(universe)-1, nonneg=True)#+self.kelly))

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
            self.sigma_squares = HistoricalVariance(kelly=True) #None
        #self.zeroforcash = True
        #self.kelly = True
        # self.standard_deviations = ParameterEstimator(standard_deviations)
        
    def _pre_evaluation(self, universe, backtest_times):
        self.sigmas_parameter = cp.Parameter(len(universe)-1) #+self.kelly))

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""
        #super()._recursive_values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)
        
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
    
    :math:`F F^T + \mathbf{diag}(d)`
    
    where :math:`F` is a *tall* matrix (many more rows than columns) and the vector
    :math:`d` is all non-negative. 
    
    :param F: exposure matrices either constant or varying in time; if so, use a pandas multiindexed
         dataframe. If None it will be fitted.
    :type F: pd.DataFrame or None
    :param d: idyosyncratic variances either constant or varying in time; If None it will be fitted.
    :type d: pd.Series or pd.DataFrame or None
    :param num_factors: number of factors (columns of F), used if fitting the model
    :type num_factors: int    
    """

    # Args:
    #     exposures (pd.DataFrame): constant factor exposure matrix or a dataframe
    #         where the first index is time.
    #     idyosync (pd.DataFrame or pd.Series): idyosyncratic variances for the symbol,
    #         either fixed (pd.Series) or through time (pd.DataFrame).
    #     factor_Sigma (pd.DataFrame or None): a constant factor covariance matrix
    #         or a DataFrame with multiindex where the first index is time. If None,
    #         the default, it is understood that the factor covariance is the identity.
    #         (Otherwise we compute its matrix square root at each step internally and
    #          apply it to the exposures).
    #     forecast_error_kappa (float or pd.Series): uncertainty on the
    #         assets' correlations. See the paper, pages 32-33.

    # """

    factor_Sigma = None

    def __init__(self, F=None, d=None, num_factors=1, kelly=True):#, normalize=False):
        self.F = F if F is None else DataEstimator(F, compile_parameter=True) 
        self.d = d if d is None else DataEstimator(d) 
        if (self.F is None) or (self.d is None):
            self.fit = True
            self.Sigma = HistoricalFactorizedCovariance(kelly=kelly) #Sigma
        else:
            self.fit = False
        self.num_factors = num_factors
        # self.kelly = True
        # self.zeroforcash = True
        # self.normalize = normalize

    # @staticmethod
    # def build_low_rank_model(rets, num_factors=10, iters=10, normalize=True, shrink=True):
    #     r"""Build a low rank risk model from past returns that include NaNs.
    #
    #     This is an experimental procedure that may work well on past returns
    #     matrices with few NaN values (say, below 20% of the total entries).
    #     If there are (many) NaNs, one should probably also use a rather
    #     large risk forecast error.
    #     """
    #     # rets = past_returns.iloc[:,:-1] # drop cash
    #     nan_fraction = rets.isnull().sum().sum() / np.prod(rets.shape)
    #     normalizer = np.sqrt((rets**2).mean())
    #     if normalize:
    #         normalized = rets/(normalizer + 1E-8)
    #     else:
    #         normalized = rets
    #     if nan_fraction:
    #         if nan_fraction > 0.1 and not shrink:
    #             warnings.warn("Low rank model estimation on past returns with many NaNs should use the `shrink` option")
    #         nan_implicit_imputation = pd.DataFrame(0., columns=normalized.columns, index = normalized.index)
    #         for i in range(iters):
    #             u, s, v = np.linalg.svd(normalized.fillna(nan_implicit_imputation), full_matrices=False)
    #             nan_implicit_imputation = pd.DataFrame(
    #                 (u[:, :num_factors] * (s[:num_factors] - s[num_factors] * shrink)) @ v[:num_factors],
    #                 columns = normalized.columns, index = normalized.index)
    #     else:
    #         u, s, v = np.linalg.svd(normalized, full_matrices=False)
    #     F = v[:num_factors].T * s[:num_factors] / np.sqrt(len(rets))
    #     if normalize:
    #         F = pd.DataFrame(F.T * (normalizer.values + 1E-8), columns=normalizer.index)
    #     else:
    #         F = pd.DataFrame(F.T, columns=normalizer.index)
    #     idyosyncratic = normalizer**2 - (F**2).sum(0)
    #     if not np.all(idyosyncratic >= 0.):
    #         raise ForeCastError("Low rank risk estimation with iterative SVD did not work.")
    #     return F, idyosyncratic

    def _pre_evaluation(self, universe, backtest_times):
        # super()._recursive_pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        self.idyosync_sqrt_parameter = cp.Parameter(len(universe)-1)
        self.F_parameter = cp.Parameter((self.num_factors, len(universe)-1)) if self.F is None else self.F.parameter
        # if not (self.factor_Sigma is None):
        #     self.factor_Sigma_sqrt = cp.Parameter(self.factor_Sigma.shape, PSD=True)
        # self.forecast_error_penalizer = cp.Parameter(returns.shape[1], nonneg=True)

    def _values_in_time(self, t, past_returns, **kwargs):
        
        # if self.F is None:
        #     if not self.kelly:
        #         past_returns = past_returns - past_returns.mean()
        #     if self.zeroforcash:
        #         past_returns = pd.DataFrame(past_returns, copy=True)
        #         past_returns.iloc[:, -1] = 0.
        #     F, d = self.build_low_rank_model(past_returns, num_factors=self.num_factors, normalize=self.normalize)
        #     self.F_parameter.value = F.values
        #     d = d.values
        # else:
        #     d = self.d.current_value
        if self.fit:
            Sigmasqrt = self.Sigma.current_value
            # numpy eigendecomposition has largest eigenvalues last
            self.F_parameter.value = Sigmasqrt[:, -self.num_factors:].T
            d = (Sigmasqrt[:, :-self.num_factors]**2).sum(1)
        else:
            d = self.d.current_value
        self.idyosync_sqrt_parameter.value = np.sqrt(d)


    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        self.expression = cp.sum_squares(cp.multiply(self.idyosync_sqrt_parameter, w_plus_minus_w_bm[:-1]))
        assert self.expression.is_dcp(dpp=True)

        self.expression += cp.sum_squares(self.F_parameter @ w_plus_minus_w_bm[:-1])
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
