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

import logging

import cvxpy as cvx
import numpy as np
import pandas as pd

from .costs import BaseCost
from .utils import values_in_time

logger = logging.getLogger(__name__)

from .estimator import ParameterEstimator

__all__ = [
    "FullCovariance",
    "RollingWindowFullCovariance",
    "ExponentialWindowFullCovariance",
    "DiagonalCovariance",
    "RollingWindowDiagonalCovariance",
    "ExponentialWindowDiagonalCovariance",
    "LowRankRollingRisk",
    "WorstCaseRisk",
    "RobustFactorModelSigma",
    "RobustSigma",
    "FactorModelSigma",
]


class BaseRiskModel(BaseCost):

    benchmark_weights = None
    
    ## DEPRECATED,BENCHMARK WEIGHTS ARE NOW PASSED BY set_benchmark
    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)
        self.benchmark_weights = self.w_bench
        # super(BaseRiskModel, self).__init__()
        # self.gamma_half_life = kwargs.pop("gamma_half_life", np.inf)
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        if self.benchmark_weights is None:
            bw = pd.Series(0., returns.columns)
            bw[-1] = 1.
            self.benchmark_weights = ParameterEstimator(bw)
        super().pre_evaluation(returns, volumes, start_time, end_time)
        
    def set_benchmark(self, benchmark_weights):
        self.benchmark_weights = ParameterEstimator(benchmark_weights)

    def weight_expr(self, t, w_plus, z, value):
        """Temporary placeholder while migrating to new interface"""
        self.expression, _ = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.expression, []

    ## DEPRECATED, MAYBE INCLUDE ITS LOGIC AT BASECOST LEVEL
    def optimization_log(self, t):
        if self.expression.value:
            return self.expression.value
        else:
            return np.NaN
            


class FullCovariance(BaseRiskModel):
    """Quadratic risk model with full covariance matrix.

    Args:
        Sigma (pd.DataFrame): DataFrame of Sigma matrices as understood 
        by `cvxportfolio.estimator.DataEstimator`.

    """

    def __init__(self, Sigma = None, **kwargs):
        ## DEPRECATED, IT'S USED BY SOME OLD CVXPORTFOLIO PIECES
        super(FullCovariance, self).__init__(**kwargs)
        self.Sigma = ParameterEstimator(Sigma, positive_semi_definite=True)

    def compile_to_cvxpy(self, w_plus, z, value):
        return cvx.quad_form(w_plus - self.benchmark_weights, self.Sigma)
    
        
class RollingWindowFullCovariance(FullCovariance):
    """Build FullCovariance model automatically with pandas rolling window.
    
    At the start of a backtest receives a view of asset returns
    (excluding cash) and computes the rolling window covariance
    for given lookback_period (default 250).
    
    Args:
        loockback_period (int): how many past returns are used each
            trading day to compute the historical covariance.
    """
    def __init__(self, lookback_period=250, zero_cash_covariance=True):
        self.lookback_period = lookback_period
        self.zero_cash_covariance = zero_cash_covariance
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        """Function to initialize object with full prescience."""
        # drop cash return
        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.
        self.Sigma = ParameterEstimator(
            # shift forward so only past returns are used
            returns.rolling(window=self.lookback_period).cov().shift(returns.shape[1]), 
            positive_semi_definite=True
        )
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time)
        

class ExponentialWindowFullCovariance(FullCovariance):
    """Build FullCovariance model automatically with pandas exponential window.
    
    At the start of a backtest receives a view of asset returns
    (excluding cash) and computes the exponential window covariance
    for given half_life (default 250).
    
    Args:
        half_life (int): the half life of exponential decay used by pandas
            exponential moving window
    """
    def __init__(self, half_life=250, zero_cash_covariance=True):
        self.half_life = half_life
        self.zero_cash_covariance = zero_cash_covariance
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        """Function to initialize object with full prescience."""

        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.
        self.Sigma = ParameterEstimator(
             # shift forward so only past returns are used
            returns.ewm(halflife=self.half_life).cov().shift(returns.shape[1]),
            positive_semi_definite=True
        )
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time)
    
        
class DiagonalCovariance(BaseRiskModel):
    """Risk model using diagonal covariance matrix.
    
    Args:
        standard_deviations (pd.DataFrame or pd.Series): per-stock standard
            deviations, defined as square roots of covariances. 
            Indexed by time if DataFrame.
    """
    
    def __init__(self, standard_deviations = None):
        self.standard_deviations = ParameterEstimator(standard_deviations)

    def compile_to_cvxpy(self, w_plus, z, value):        
        return cvx.sum_squares(cvx.multiply(w_plus, self.standard_deviations))
        
        
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
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        """Function to initialize object with full prescience."""
        # drop cash return
        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.
        self.standard_deviations = ParameterEstimator(
            # shift forward so only past returns are used
            np.sqrt(returns.rolling(window=self.lookback_period).var().shift(1), 
        ))
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time)
        

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
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        """Function to initialize object with full prescience."""

        # drop cash return
        if self.zero_cash_covariance:
            returns = returns.copy(deep=True)
            returns.iloc[:, -1] = 0.
        self.standard_deviations = ParameterEstimator(
             # shift forward so only past returns are used
            np.sqrt(returns.ewm(halflife=self.half_life).var().shift(1),
        ))
        # initialize cvxpy Parameter(s)
        super().pre_evaluation(returns, volumes, start_time, end_time)
        
        
class FactorModelSigma(BaseRiskModel):
    def __init__(self, exposures, factor_Sigma, idiosync, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""

        self.exposures = ParameterEstimator(exposures)
        self.factor_Sigma = ParameterEstimator(factor_Sigma)
        self.idiosync = ParameterEstimator(idiosync)
        # DEPRECATED BUT USED BY OLD CVXPORTFOLIO
        # super(FactorModelSigma, self).__init__(**kwargs)

    def compile_to_cvxpy(self, w_plus, z, value):
        self.expression = cvx.sum_squares(
            cvx.multiply(np.sqrt(self.idiosync), wplus)
        ) + cvx.quad_form(
            (wplus.T @ self.exposures.T).T,
            self.factor_Sigma,
        )
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

    def __init__(self, lookback, zero_cash_risk=True):
        self.lookback = lookback
        self.zero_cash_risk = zero_cash_risk
        
    def pre_evaluation(self, returns, volumes, start_time, end_time):
        """Function to initialize object with full prescience."""
        self.recent_returns = cvx.Parameter(shape=(self.lookback, returns.shape[1]))
        super().pre_evaluation(returns, volumes, start_time, end_time)
        
    def values_in_time(self, t, past_returns, *args, **kwargs):
        val = past_returns.iloc[-self.lookback:].copy(deep=True)
        if self.zero_cash_risk:
            val.iloc[:, -1] = 0.
        self.recent_returns.value = val.values
        # update attributes
        super().values_in_time(t, past_returns=past_returns, *args, **kwargs)
    
    def compile_to_cvxpy(self, w_plus, z, value):
        return cvx.sum_squares(self.recent_returns @ (w_plus - self.benchmark_weights)) / self.lookback

    # def _estimate(self, t, wplus, z, value):
    #     idx = self.returns.index.get_loc(t)
    #     # TODO make sure pandas + cvxpy works
    #     R = self.returns.iloc[max(idx - 1 - self.lookback, 0) : idx - 1]
    #     assert R.shape[0] > 0
    #     self.expression = cvx.sum_squares(R.values * wplus) / self.lookback
    #     return self.expression


class RobustSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""

    def __init__(self, Sigma, epsilon, **kwargs):
        self.Sigma = Sigma  # pd.Panel or matrix
        self.epsilon = epsilon  # pd.Series or scalar
        super(RobustSigma, self).__init__(**kwargs)

    def _estimate(self, t, w_plus, z, value):
        self.expression = (
            cvx.quad_form(wplus, values_in_time(self.Sigma, t))
            + values_in_time(self.epsilon, t)
            * (cvx.abs(w_plus).T * np.diag(values_in_time(self.Sigma, t))) ** 2
        )

        return self.expression


class RobustFactorModelSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""

    def __init__(self, exposures, factor_Sigma, idiosync, epsilon, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""
        self.exposures = exposures
        assert not exposures.isnull().values.any()
        self.factor_Sigma = factor_Sigma
        assert not factor_Sigma.isnull().values.any()
        self.idiosync = idiosync
        assert not idiosync.isnull().values.any()
        self.epsilon = epsilon
        super(RobustFactorModelSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        F = values_in_time(self.exposures, t)
        f = (wplus.T * F.T).T
        Sigma_F = values_in_time(self.factor_Sigma, t)
        D = values_in_time(self.idiosync, t)
        self.expression = (
            cvx.sum_squares(cvx.multiply(np.sqrt(D), wplus))
            + cvx.quad_form(f, Sigma_F)
            + self.epsilon * (cvx.abs(f).T * np.sqrt(np.diag(Sigma_F))) ** 2
        )

        return self.expression


class WorstCaseRisk(BaseRiskModel):
    def __init__(self, riskmodels, **kwargs):
        self.riskmodels = riskmodels
        super(WorstCaseRisk, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        self.risks = [risk.weight_expr(t, wplus, z, value) for risk in self.riskmodels]
        return cvx.max_elemwise(*self.risks)

    def optimization_log(self, t):
        """Return data to log in the result object."""
        return pd.Series(
            index=[model.__class__.__name__ for model in self.riskmodels],
            data=[risk.value[0, 0] for risk in self.risks],
        )
