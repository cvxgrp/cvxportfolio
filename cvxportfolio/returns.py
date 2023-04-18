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
"""This module contains classes that define return models for
portfolio optimization policies, and related objects.
"""

import cvxpy as cvx

# from cvxportfolio.expression import Expression
from .utils import values_in_time, null_checker
from .costs import BaseCost

from .estimator import DataEstimator, ParameterEstimator

__all__ = ["ReturnsForecast", 
           "RollingWindowReturnsForecast",
           "ExponentialWindowReturnsForecast",
           "LegacyReturnsForecast",
           "MPOReturnsForecast", 
           "MultipleReturnsForecasts"]


class BaseReturnsModel(BaseCost):
    """Base class for return models.
    
    Use this to define any logic common to return models.
    """
    pass



class ReturnsForecast(BaseReturnsModel):
    """Simple return forecast provided by the user.

    Args:
        expected_returns (pd.DataFrame or pd.Series): constant per-symbol
            forecasts (if Series) or varying in time (if DataFrame)
    """
    
    def __init__(self, expected_returns):
        self.expected_returns = ParameterEstimator(expected_returns)
        
    def compile_to_cvxpy(self, w_plus, z, v):
        return w_plus.T @ self.expected_returns


class RollingWindowReturnsForecast(ReturnsForecast):
    """Compute returns forecast by rolling window mean of past returns.
    
    Args:
        lookback_period (int): how many past returns are used at each point in time.
            Default is 250.
        use_last_for_cash (bool): for the cash return instead just use the last
            value. Default True.
    """
    
    def __init__(self, lookback_period, use_last_for_cash=True):
        self.lookback_period = lookback_period
        self.use_last_for_cash = use_last_for_cash
        
    def pre_evaluation(self, returns, volumes, start_time, end_time,  **kwargs):
        forecasts = returns.rolling(window=self.lookback_period).mean().shift(1)
        if self.use_last_for_cash:
            forecasts.iloc[:, -1] = returns.iloc[:, -1].shift(1)
        self.expected_returns = ParameterEstimator(forecasts)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)
        
        
class ExponentialWindowReturnsForecast(ReturnsForecast):
    """Compute returns forecast by exponential window mean of past returns.
    
    Args:
        half_life (int): Half life of exponential decay used. Default is 250.
        use_last_for_cash (bool): for the cash return instead just use the last
            value. Default True.
    """
    
    def __init__(self, half_life, use_last_for_cash=True):
        self.half_life = half_life
        self.use_last_for_cash = use_last_for_cash
        
    def pre_evaluation(self, returns, volumes, start_time, end_time,  **kwargs):
        forecasts = returns.ewm(halflife=self.half_life).mean().shift(1)
        if self.use_last_for_cash:
            forecasts.iloc[:, -1] = returns.iloc[:, -1].shift(1)
        self.expected_returns = ParameterEstimator(forecasts)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)

class LegacyReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    STILL USED BY OLD PARTS OF CVXPORTFOLIO

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0.0, gamma_decay=None, name=None):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = cvx.multiply(values_in_time(self.returns, t), wplus)
        alpha -= cvx.multiply(values_in_time(self.delta, t), cvx.abs(wplus))
        return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data):
        self.alpha_data = alpha_data

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        return self.alpha_data[(t, tau)].values.T * wplus


class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    DEPRECATED: THIS SHOULD BE DONE BY MULTIPLYING BY HYPERPARAMETERS
    AND PASSING MULTIPLE RETURN MODELS LIKE WE DO FOR COSTS

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr_ahead(t, tau, wplus) * self.weights[idx]
        return alpha
