# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
# Copyright 2023- The Cvxportfolio Contributors
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

from .estimator import ParameterEstimator, DataEstimator  # , ConstantEstimator
import logging

import scipy.linalg

import cvxpy as cvx
import numpy as np
import pandas as pd

from .costs import BaseCost
# from .utils import values_in_time

logger = logging.getLogger(__name__)


__all__ = [
    "FullCovariance",
    #"RollingWindowFullCovariance",
    #"ExponentialWindowFullCovariance",
    "DiagonalCovariance",
    "RollingWindowDiagonalCovariance",
    "ExponentialWindowDiagonalCovariance",
    "LowRankRollingRisk",
    "RollingWindowFactorModelRisk",
    "WorstCaseRisk",
    "FactorModelRisk",
    # "CashBenchMark",
]


# class CashBenchMark(ConstantEstimator):
#     """Default benchmark weights for cvxportfolio risk models.
#     """
#
#     def __init__(self):
#         pass
#
#     def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
#         """Initialize it using the size of the returns.
#         """
#         size = returns.shape[1]
#         value = np.zeros(size)
#         value[-1] = 1.
#         super().__init__(value)
#         super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class BaseRiskModel(BaseCost):
    # benchmark_weights = None

    # DEPRECATED,BENCHMARK WEIGHTS ARE NOW PASSED BY set_benchmark
    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)
        self.benchmark_weights = None #self.w_bench
        # super(BaseRiskModel, self).__init__()
        # self.gamma_half_life = kwargs.pop("gamma_half_life", np.inf)

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        if not hasattr(self, 'benchmark_weights') or self.benchmark_weights is None:
            bw = pd.Series(0.0, returns.columns)
            bw.iloc[-1] = 1.0
            self.benchmark_weights = bw  # ParameterEstimator(bw)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

    def set_benchmark(self, benchmark_weights):
        """We can only have constant benchmark because otherwise it is not dpp compliant.

        DEPRECATED: IT SHOULD NOT BE PASSED HERE. IT SHOULD BE PASSED TO POLICY
            AND ADDED w_plus_wrt_bm as a variable with equality constraint.
        """
        self.benchmark_weights = benchmark_weights  # ParameterEstimator(benchmark_weights)

    def weight_expr(self, t, w_plus, z, value):
        """Temporary placeholder while migrating to new interface"""
        self.expression, _ = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.expression, []

    # DEPRECATED, MAYBE INCLUDE ITS LOGIC AT BASECOST LEVEL
    def optimization_log(self, t):
        if self.expression.value:
            return self.expression.value
        else:
            return np.NaN


class FullCovariance(BaseRiskModel):
    r"""Quadratic risk model with full covariance matrix.
    
    This class represents the term :math:`\Sigma_t`, *i.e.,*
    the :math:`(n-1) \times (n-1)` positive semi-definite matrix
    which estimates the covariance of the (non-cash) assets' returns.
    :ref:`Optimization-based policies` use this, as is explained 
    in Chapter 4 and 5 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    
    The user can either supply a :class:`pandas.DataFrame` with the covariance matrix
    (constant or varying in time) computed externally (for example
    with some machine learning technique) or let this class estimate the covariance from the data. 
    The latter is the default behavior.
    
    This class implements three ways to compute the covariance matrix from the past returns. The
    computation is repeated at each point in time :math:`t` of a :class:`BackTest` using only
    the past returns available at that point: :math:`r_{t-1}, r_{t-2}, \ldots`.
    
    * *rolling covariance*, using :class:`pandas.DataFrame.rolling.cov`. This is done
      if the user specifies the ``rolling`` argument.
    * *exponential moving window covariance*, using :class:`pandas.DataFrame.ewm.cov`. This is done
      if the user specifies the ``halflife`` argument (``rolling`` takes precedence).
    * *full historical covariance*, using :class:`pandas.DataFrame.cov`. This is the default
      behavior if no arguments are specified.
    
    If there are missing data in the historical returns the estimated covariance may not
    be positive semi-definite. We correct it by projecting on the positive semi-definite 
    cone (*i.e.*, we set the negative eigenvalues of the resulting :math:`\Sigma_t` to zero). 

    :param Sigma: :class:`pandas.DataFrame` of covariance matrices 
        supplied by the user. The DataFrame either represents a single (constant) covariance matrix
        or one for each point in time. In the latter case the DataFrame must have a :class:`pandas.MultiIndex`
        where the first level is a :class:`pandas.DatetimeIndex`. If ``None`` (the default)
        the covariance matrix is computed from past returns.
    :type Sigma: pandas.DataFrame or None
    :param rolling: if it is not ``None`` the covariance matrix will be estimated
        on a rolling window of size ``rolling`` of the past returns.
    :type rolling: int or None
    :param halflife: if it is not ``None`` the covariance matrix will be estimated
        on an exponential moving window of the past returns with half-life ``halflife``. 
        If ``rolling`` is specified it takes precedence over ``halflife``. If both are ``None`` the full history 
        will be used for estimation.
    :type halflife: int or None
    :param kappa: the multiplier for the associated forecast error risk 
        (see pages 32-33 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_).
        If ``float`` a passed it is treated as a constant, if ``pandas.Series`` with ``pandas.DateTime`` index
        it varies in time, if ``None`` the forecast error risk term will not be compiled.
    :type kappa: float or pandas.Series or None
    :param addmean: correct the covariance matrix with the term :math:`\mu\mu^T`, as is explained
        in page 28 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_, 
        to match the second term of the Taylor expansion of the portfolio log-return. Default
        is ``False``, corresponding to classical mean-variance optimization. If ``True``, it 
        estimates :math:`\mu` with the same technique as :math:`\Sigma`, *i.e.*, with rolling window
        average, exponential moving window average, or an average of the full history.
    :type addmean: bool
    """

    def __init__(self, Sigma=None, rolling=None, halflife=None, kappa=None, addmean=False):
        if not Sigma is None:
            self.Sigma = ParameterEstimator(Sigma, positive_semi_definite=True)
        else:
            self.Sigma = None
        self.rolling = rolling if Sigma is None else None
        self.halflife = halflife if Sigma is None else None
        self.full = self.Sigma is None and self.rolling is None and self.halflife is None
        if not kappa is None:
            self.kappa = DataEstimator(kappa)
        else:
            self.kappa = None
        self.addmean = addmean
        if self.addmean:
            raise NotImplementedError
        # self.forecast_error_kappa = DataEstimator(
        #     forecast_error_kappa)  # ParameterEstimator(
        # self.parameter_forecast_error = cvx.Parameter(
        #     Sigma.shape[1], nonneg=True)
        #    forecast_error_kappa, non_negative=True
        # )

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        
        if not self.kappa is None:
            self.forecast_error = cvx.Parameter(returns.shape[1]-1, nonneg=True)

        if not self.rolling is None:
            forecasts = returns.iloc[:, :-1].rolling(window=self.rolling).cov().shift(
                returns.shape[1]-1)
                
        elif not self.halflife is None:
            forecasts = returns.iloc[:, :-1].ewm(halflife=self.halflife).cov().shift(
                returns.shape[1]-1)
                
        elif self.full:
            self.Sigma_sqrt = cvx.Parameter((returns.shape[1]-1, returns.shape[1]-1))
            return
        
        if self.Sigma is None:
            self.Sigma = ParameterEstimator(forecasts)
        
        self.Sigma_sqrt = cvx.Parameter((returns.shape[1]-1, returns.shape[1]-1))
        
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        

    def values_in_time(self, t, current_weights, current_portfolio_value, past_returns, past_volumes,
            **kwargs):
        """Update forecast error risk here, and take square root of Sigma."""
        super().values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)
        
        if not self.full:
            current_Sigma = self.Sigma.value
        else:
            current_Sigma = past_returns.iloc[:,:-1].cov()
            
        eigval, eigvec = np.linalg.eigh(current_Sigma)
        eigval = np.maximum(eigval, 0.)
        self.Sigma_sqrt.value = eigvec @ np.diag(np.sqrt(eigval))
        
        if not self.kappa is None:
            self.forecast_error.value = np.sqrt(np.diag(self.Sigma_sqrt.value @ self.Sigma_sqrt.value.T)) * \
                np.sqrt(self.forecast_error_kappa.current_value)
        
        
        #self.parameter_forecast_error.value = np.sqrt(
        #    np.diag(self.Sigma.value)) * np.sqrt(self.forecast_error_kappa.current_value)
        #if not self.LEGACY:
        #self.Sigma_sqrt.value = scipy.linalg.sqrtm(self.Sigma.value)
        # assert np.allclose(
        #     self.Sigma.value,
        #     self.Sigma_sqrt.value @ self.Sigma_sqrt.value.T)

    def compile_to_cvxpy(self, w_plus, z, value):
        
        self.cvxpy_expression = cvx.sum_squares(self.Sigma_sqrt.T @ (w_plus - self.benchmark_weights)[:-1])
    
        if not self.kappa is None:
            self.cvxpy_expression += cvx.square(cvx.abs(w_plus - self.benchmark_weights)[:-1].T @ self.forecast_error)
            
        # assert self.cvxpy_expression.is_dcp(dpp=True)
        return self.cvxpy_expression


# class RollingWindowFullCovariance(FullCovariance):
#     """Build FullCovariance model automatically with pandas rolling window.
#
#     At the start of a backtest receives a view of asset returns
#     (excluding cash) and computes the rolling window covariance
#     for given lookback_period (default 250).
#
#     Args:
#         loockback_period (int): how many past returns are used each
#             trading day to compute the historical covariance.
#     """
#
#     def __init__(
#             self,
#             lookback_period=250,
#             zero_cash_covariance=True,
#             forecast_error_kappa=0.0):
#         self.lookback_period = lookback_period
#         self.zero_cash_covariance = zero_cash_covariance
#         self.forecast_error_kappa = DataEstimator(forecast_error_kappa)
#
#     def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
#         """Function to initialize object with full prescience."""
#         # drop cash return
#         if self.zero_cash_covariance:
#             returns = returns.copy(deep=True)
#             returns.iloc[:, -1] = 0.0
#         self.Sigma = ParameterEstimator(
#             # shift forward so only past returns are used
#             returns.rolling(
#                 window=self.lookback_period).cov().shift(
#                 returns.shape[1]),
#             positive_semi_definite=True,
#         )
#         # initialize cvxpy Parameter(s)
#         self.parameter_forecast_error = cvx.Parameter(
#             returns.shape[1], nonneg=True)
#
#         super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
#
#
# class ExponentialWindowFullCovariance(FullCovariance):
#     """Build FullCovariance model automatically with pandas exponential window.
#
#     At the start of a backtest receives a view of asset returns
#     (excluding cash) and computes the exponential window covariance
#     for given half_life (default 250).
#
#     Args:
#         half_life (int): the half life of exponential decay used by pandas
#             exponential moving window
#     """
#
#     def __init__(
#             self,
#             half_life=250,
#             zero_cash_covariance=True,
#             forecast_error_kappa=0.0):
#         self.half_life = half_life
#         self.zero_cash_covariance = zero_cash_covariance
#         self.forecast_error_kappa = DataEstimator(forecast_error_kappa)
#
#     def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
#         """Function to initialize object with full prescience."""
#
#         if self.zero_cash_covariance:
#             returns = returns.copy(deep=True)
#             returns.iloc[:, -1] = 0.0
#         self.Sigma = ParameterEstimator(
#             # shift forward so only past returns are used
#             returns.ewm(halflife=self.half_life).cov().shift(returns.shape[1]),
#             positive_semi_definite=True,
#         )
#         self.parameter_forecast_error = cvx.Parameter(
#             returns.shape[1], nonneg=True)
#         # initialize cvxpy Parameter(s)
#         super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class DiagonalCovariance(BaseRiskModel):
    """Risk model using diagonal covariance matrix.

    Args:
        standard_deviations (pd.DataFrame or pd.Series): per-stock standard
            deviations, defined as square roots of covariances.
            Indexed by time if DataFrame.
    """

    def __init__(self, standard_deviations=None):
        self.standard_deviations = ParameterEstimator(standard_deviations)

    def compile_to_cvxpy(self, w_plus, z, value):
        return cvx.sum_squares(
            cvx.multiply(
                w_plus -
                self.benchmark_weights,
                self.standard_deviations))


class RollingWindowDiagonalCovariance(DiagonalCovariance):
    """Build DiagonalCovariance model automatically with pandas rolling window.

    At the start of a backtest receives a view of asset returns
    (excluding cash) and computes the rolling window variances
    for given lookback_period (default 250).

    Args:
        loockback_period (int): how many past returns are used each
            trading day to compute the historical covariance.
    """

    def __init__(self, lookback_period=250, zero_cash_covariance=True):
        self.lookback_period = lookback_period
        self.zero_cash_covariance = zero_cash_covariance

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Function to initialize object with full prescience."""
        # drop cash return
        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.0
        self.standard_deviations = ParameterEstimator(
            # shift forward so only past returns are used
            np.sqrt(
                returns.rolling(window=self.lookback_period).var().shift(1),
            )
        )
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class ExponentialWindowDiagonalCovariance(DiagonalCovariance):
    """Build DiagonalCovariance model automatically with pandas exponential window.

    At the start of a backtest receives a view of asset returns
    (excluding cash) and computes the exponential window variances
    for given half_life (default 250).

    Args:
        half_life (int): the half life of exponential decay used by pandas
            exponential moving window
    """

    def __init__(self, half_life=250, zero_cash_covariance=True):
        self.half_life = half_life
        self.zero_cash_covariance = zero_cash_covariance

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Function to initialize object with full prescience."""

        # drop cash return
        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.0
        self.standard_deviations = ParameterEstimator(
            # shift forward so only past returns are used
            np.sqrt(
                returns.ewm(halflife=self.half_life).var().shift(1),
            )
        )
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class FactorModelRisk(BaseRiskModel):
    """Factor model covariance.

    Args:
        exposures (pd.DataFrame): constant factor exposure matrix or a dataframe
            where the first index is time.
        idyosync (pd.DataFrame or pd.Series): idyosyncratic variances for the symbol,
            either fixed (pd.Series) or through time (pd.DataFrame).
        factor_Sigma (pd.DataFrame or None): a constant factor covariance matrix
            or a DataFrame with multiindex where the first index is time. If None,
            the default, it is understood that the factor covariance is the identity.
            (Otherwise we compute its matrix square root at each step internally and
             apply it to the exposures).
        forecast_error_kappa (float or pd.Series): uncertainty on the
            assets' correlations. See the paper, pages 32-33.

    """

    factor_Sigma = None

    def __init__(self, exposures, idyosync, factor_Sigma=None, forecast_error_kappa=0.0, use_last_available_time=False, **kwargs):
        if not (factor_Sigma is None):
            self.factor_Sigma = DataEstimator(factor_Sigma, use_last_available_time=True)
            # we copy the exposures because we'll modify them
            assert isinstance(exposures, pd.DataFrame)
            exposures = pd.DataFrame(exposures, copy = True)
        else:
            self.factor_Sigma = None
        self.exposures = ParameterEstimator(exposures, use_last_available_time=use_last_available_time)
        self.idyosync = ParameterEstimator(idyosync, use_last_available_time=use_last_available_time)
        self.forecast_error_kappa = forecast_error_kappa
        # if ((np.isscalar(forecast_error_kappa) and forecast_error_kappa > 0)
        #     or (np.any(forecast_error_kappa > 0))) and factor_Sigma is not None:
        #     raise NotImplementedError("You should do Cholesky decompositions of the factor_Sigmas"
        #     "and apply them to the exposures.")

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        self.idyosync_sqrt = cvx.Parameter(returns.shape[1])
        # if not (self.factor_Sigma is None):
        #     self.factor_Sigma_sqrt = cvx.Parameter(self.factor_Sigma.shape, PSD=True)
        self.forecast_error_penalizer = cvx.Parameter(returns.shape[1], nonneg=True)

    def values_in_time(self, t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs):
        super().values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)
        self.idyosync_sqrt.value = np.sqrt(self.idyosync.value)
        if not (self.factor_Sigma is None):
            factor_Sigma_sqrt = scipy.linalg.sqrtm(self.factor_Sigma.current_value)
            assert np.allclose(self.factor_Sigma.current_value, factor_Sigma_sqrt @ factor_Sigma_sqrt.T)
            self.exposures.value = factor_Sigma_sqrt @ self.exposures.value
        self.forecast_error_penalizer.value = np.sqrt(np.sum(self.exposures.value**2, axis=0) + self.idyosync.value)

    def compile_to_cvxpy(self, w_plus, z, value):
        self.expression = cvx.sum_squares(cvx.multiply(self.idyosync_sqrt, (w_plus - self.benchmark_weights)))
        assert self.expression.is_dcp(dpp=True)
        # if not (self.factor_Sigma is None):
        #     self.expression += cvx.sum_squares(
        #         self.factor_Sigma_sqrt.T @ self.exposures @ (w_plus - self.benchmark_weights))
        #     # self.expression += cvx.quad_form((w_plus.T @ self.exposures.T).T, self.factor_Sigma)
        #     assert self.expression.is_dcp(dpp=True)
        # else:
        self.expression += cvx.sum_squares(self.exposures @ (w_plus - self.benchmark_weights))
        assert self.expression.is_dcp(dpp=True)

        # forecast error risk, assuming factor_Sigma is the identity
        self.expression += self.forecast_error_kappa * cvx.square(
            cvx.abs(w_plus - self.benchmark_weights).T @ self.forecast_error_penalizer
            # @ cvx.sqrt(cvx.sum(cvx.square(self.exposures), axis=0) + self.idyosync)
        )
        assert self.expression.is_dcp(dpp=True)
        return self.expression


class LowRankRollingRisk(BaseRiskModel):
    """Use `lookback` past returns to build low-rank structured risk model.

    Use this rather that RollingWindowFullCovariance when there are (many) more
     assets than lookback periods, to limit overfit and avoid allocating large matrices.
    (Or, better, use a factor model covariance).

    Args:
        lookback (int): how many past returns are used at each point in time.
            Default 250.
    """

    def __init__(self, lookback=250, zero_cash_risk=True):
        self.lookback = lookback
        self.zero_cash_risk = zero_cash_risk

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Function to initialize object with full prescience."""
        self.recent_returns = cvx.Parameter(
            shape=(self.lookback, returns.shape[1]))
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

    def values_in_time(
            self,
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs):
        val = past_returns.iloc[-self.lookback:].copy(deep=True)
        if self.zero_cash_risk:
            val.iloc[:, -1] = 0.0
        self.recent_returns.value = val.values
        # update attributes
        super().values_in_time(
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs)

    def compile_to_cvxpy(self, w_plus, z, value):
        self.expression = (cvx.sum_squares(
            self.recent_returns @ (w_plus - self.benchmark_weights)) / self.lookback)
        assert self.expression.is_dcp(dpp=True)
        return self.expression


class RollingWindowFactorModelRisk(FactorModelRisk):
    """Build FactorModelRisk model automatically with rolling window and svd.

    Procedure is detailed in the paper, pages 59-60, although here it is simplified.

    Eventually this will also have logic to store the computed exposures
    and idyosyncratic variances.

    Args:
        lookback (int): how many past returns are used at each point in time
            to estimate the risk model. Default 250.
        num_factors (int): how many factors in the risk model. Default is 1.
        zero_cash_risk (bool): whether to set the column and row of the (implied)
            resulting covariance matrix to zero. Default True.
    """

    def __init__(
            self,
            lookback=250,
            num_factors=1,
            zero_cash_risk=True,
            forecast_error_kappa=0.0,
            on_correlation = False):
        self.lookback = lookback
        self.num_factors = num_factors
        self.zero_cash_risk = zero_cash_risk
        self.forecast_error_kappa = forecast_error_kappa
        self.on_correlation = on_correlation

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Function to initialize object with full prescience."""

        self.idyosync = cvx.Parameter(returns.shape[1])
        self.exposures = cvx.Parameter((self.num_factors, returns.shape[1]))
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

    def values_in_time(
            self,
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs):
        val = past_returns.iloc[-self.lookback:].copy(deep=True)
        if self.zero_cash_risk:
            val.iloc[:, -1] = 0.0
        if self.on_correlation:
            stds = val.std()
            val /= stds

        total_variances = np.sum(val**2, axis=0) / self.lookback
        u, s, v = np.linalg.svd(val, full_matrices=False)

        self.exposures.value = (
            v[: self.num_factors].T * s[: self.num_factors]).T / np.sqrt(self.lookback)

        self.idyosync.value = (
            total_variances -
            np.sum(
                self.exposures.value**2,
                axis=0)).values
        assert np.all(self.idyosync.value >= 0.0)
        
        if self.on_correlation:
            self.idyosync *= stds**2
            self.exposures *= stds

        super().values_in_time(
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs)


class WorstCaseRisk(BaseRiskModel):
    """Select the most restrictive risk model for each value of the allocation vector.

    Given a list of risk models, penalize the portfolio allocation by the
    one with highest risk value at the solution point. If uncertain about
    which risk model to use this procedure can be an easy solution.

        Args:
            riskmodels (BaseRiskModel): list of BaseRiskModel classes. If using
                non-cash benchmarks, they should be set to each risk model
                individually. Calling set_benchmark on this class has no effect.
    """

    def __init__(self, riskmodels):
        self.riskmodels = riskmodels

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        """Initialize objects."""
        for risk in self.riskmodels:
            risk.pre_evaluation(
                returns,
                volumes,
                start_time,
                end_time,
                **kwargs)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

    def values_in_time(
            self,
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs):
        """Update parameters."""
        for risk in self.riskmodels:
            risk.values_in_time(
                t,
                current_weights,
                current_portfolio_value,
                past_returns,
                past_volumes,
                **kwargs)
        super().values_in_time(
            t,
            current_weights,
            current_portfolio_value,
            past_returns,
            past_volumes,
            **kwargs)

    def compile_to_cvxpy(self, w_plus, z, value):
        risks = [risk.compile_to_cvxpy(w_plus, z, value)
                 for risk in self.riskmodels]
        return cvx.max(cvx.hstack(risks))
