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
import numpy as np

# from cvxportfolio.expression import Expression
# from .legacy import values_in_time, null_checker
from .costs import BaseCost
from .risks import BaseRiskModel
from .estimator import DataEstimator, ParameterEstimator

__all__ = [
    "ReturnsForecast",
    "RollingWindowReturnsForecast",
    "ExponentialWindowReturnsForecast",
    "ReturnsForecastErrorRisk",
    "RollingWindowReturnsForecastErrorRisk",
]


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

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        forecasts = returns.rolling(
            window=self.lookback_period).mean().shift(1)
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

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        forecasts = returns.ewm(halflife=self.half_life).mean().shift(1)
        if self.use_last_for_cash:
            forecasts.iloc[:, -1] = returns.iloc[:, -1].shift(1)
        self.expected_returns = ParameterEstimator(forecasts)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class ReturnsForecastErrorRisk(BaseRiskModel):
    """Simple return forecast error risk with values provided by the user.

    Implements the model described in pages 31-32 of the paper. You
    must pass the delta Series (if constant) or DataFrame (if time-varying)
    of the forecast errors. Or, use one of the classes below to
    compute it automatically. Multiply this object by an external multiplier
    (which can itself be a ParameterEstimator) and calibrate for the right
    amount of penalization.

    Args:
        deltas_errors (pd.DataFrame or pd.Series): constant per-symbol
            errors on the returns forecasts (if Series),
             or varying in time (if DataFrame).
    """

    def __init__(self, deltas_errors):
        self.deltas_errors = ParameterEstimator(deltas_errors)

    def compile_to_cvxpy(self, w_plus, z, v):
        return cvx.abs(w_plus - self.benchmark_weights).T @ self.deltas_errors


class RollingWindowReturnsForecastErrorRisk(ReturnsForecastErrorRisk):
    """Compute returns forecast errors with rolling window of past returns.

    We compute the forecast error as the standard deviation of the mean
    estimator on a rolling window. That is, the rolling window standard deviation
    divided by sqrt(lookback_period), i.e., the square root of the number of samples
    used.

    Args:
        lookback_period (int): how many past returns are used at each point in time.
            Default is 250.
        zero_for_cash (bool): for the cash return forecast instead the error is zero.
    """

    def __init__(self, lookback_period, zero_for_cash=True):
        self.lookback_period = lookback_period
        self.zero_for_cash = zero_for_cash

    def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
        tmp = returns.rolling(window=self.lookback_period).std().shift(
            1) / np.sqrt(self.lookback_period)
        if self.zero_for_cash:
            tmp.iloc[:, -1] = 0.0
        self.deltas_errors = ParameterEstimator(tmp)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class ExponentialWindowReturnsForecastErrorRisk(ReturnsForecastErrorRisk):
    """Compute returns forecast errors with exponential window of past returns.

    Currently not implemented; we need to work out the math. It's probably
    as simple as ewm(...).std() / np.sqrt(half_life).
    """

    def __init__(self):
        raise NotImplementedError


