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
from .costs import BaseCost, CombinedCosts
from .risks import BaseRiskModel
from .estimator import DataEstimator, ParameterEstimator

__all__ = [
    "ReturnsForecast",
    "MultipleReturnsForecasts",
    "RollingWindowReturnsForecast",
    "ExponentialWindowReturnsForecast",
    "ReturnsForecastErrorRisk",
    "RollingWindowReturnsForecastErrorRisk",
]


class BaseReturnsModel(BaseCost):
    """Base class for return models.

    Use this to define any logic common to return models.
    """

    # interface to old
    def weight_expr(self, t, w_plus, z=None, value=None):
        cost, constr = self._estimate(t, w_plus, z, value)
        return cost


class ReturnsForecast(BaseReturnsModel):
    r"""Returns forecast, either provided by the user or computed from the data.
    
    This class represents the term :math:`\hat{r}_t`, 
    the forecast of assets' returns at time :math:`t`.
    :ref:`Optimization-based policies` use this, typically as the first
    element of their objectives.
    See Chapters 4 and 5 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_
    for more details.
    
    It can either get returns forecast from the user, for example
    computed with some machine learning technique, or it can estimate them 
    automatically from the data. 
    
    In the first case ``r_hat`` is specified as
    
    * :class:`float`, if :math:`\hat{r}_t` is the same for all times and assets
    * :class:`pandas.Series` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` is the same for all assets but changes in time
    * :class:`pandas.Series` indexed by assets' names, if :math:`\hat{r}_t` is constant in time and changes across assets
    * :class:`pandas.DataFrame` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` changes across time and assets.
    
    The returns' forecast provided by the user must be supplied for all assets
    including cash (unless it is constant across all assets, so, also cash).
    
    If instead ``r_hat`` is not speficied it defaults to None. This instructs
    this class to compute :math:`\hat{r}_t` instead. It is
    done, at each step of a :class:`BackTest`, by evaluating an average of the
    past returns (*i.e.,* the real returns :math:`{r}_{t-1}, {r}_{t-2}, \ldots`,
    where :math:`t` is the current time). This class implements three ways to do so.
    
    * The full average of all available past returns skipping :class:`numpy.nan` values. 
      This is the default mode if no parameters are passed.
    * a *rolling window*  average of recent returns. This is specified by setting
      the ``rolling`` parameter to a positive integer. If specified it takes precedence
      over ``ewm``. See the documentation of :class:`pandas.DataFrame.rolling` for
      details. It fails if the specified window contains :class:`numpy.nan` values.
    * an *exponential moving window* average of past returns. This is specified
      by setting the ``halflife`` parameter to a positive integer.
      The exponential moving window is a weighted
      average where the most recent returns have a larger weight than the older
      ones. This is done by calling :class:`pandas.DataFrame.ewm` with the
      `halflife`` argument, which represents the number of periods over which
      the weight decays by a factor of 2. See the relevant ``pandas`` documentation
      for more details.
      

    The only exception to the above is the forecast of cash return 
    :math:`{\left(\hat{r}_t\right)}_n`, for which the observed value
    from last period :math:`{\left({r}_{t-1}\right)}_n`
    is typically chosen. This is
    the default behavior and is done by setting ``lastforcash`` to ``True``.
    (If instead ``lastforcash`` is ``False``, the same averaging of past 
    returns is used to forecast the cash return as well.)
    
    :param r_hat: return forecasts supplied by the user, default is None
    :type r_hat: float or pandas.Series or pandas.DataFrame or None
    :param rolling: size of rolling window, it takes precendence over 
        ``halflife`` if both are specified
    :type rolling: int or None
    :param halflife: half-life of exponential moving window
    :type halflife: int or None
    :param lastforcash: use last value to estimate cash return 
    :type lastforcash: bool
    
    :raises cvxportfolio.MissingValuesError: If the class accesses 
        user-provided elements of ``r_hat`` that are :class:`numpy.nan`,
        or the rolling window used for averaging contains :class:`numpy.nan`.
    :raises ValueError: If the data passed by the user is not of the right
        size (for example the cash returns' columns is missing).
    
    :Example:
    
    >>> import cvxportfolio as cp
    >>> policy = cp.SinglePeriod(cp.ReturnsForecast() - \
        0.5 * cp.FullCovariance())
    
    Defines a single period optimization policy where the returns' forecasts
    :math:`\hat{r}_t` are the full average of past returns at each point in time
    and the risk model is the full covariance, also computed from the past returns.
    """    

    def __init__(self, r_hat=None, rolling=None, halflife=None, lastforcash=True):
        self.expected_returns = ParameterEstimator(r_hat)
        
        self.name = 'PLACEHOLDER'

    def compile_to_cvxpy(self, w_plus, z, v):
        return w_plus.T @ self.expected_returns
        
   
        
class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    DEPRECATED: THIS CLASS IS KEPT TO REUSE OLD TESTS
    IT'S NOW IMPLEMENTED BY COMBINED COSTS, I.E.,
    YOU JUST WRITE AN ALGEBRA OF RETURNS FORECASTS
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights
    
    def pre_evaluation(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el.pre_evaluation(*args, **kwargs) for el in self.alpha_sources]

    def values_in_time(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el.values_in_time(*args, **kwargs) for el in self.alpha_sources] 
               
    def compile_to_cvxpy(self, w_plus, z, v):
        return sum([el.compile_to_cvxpy(w_plus, z, v) * self.weights[i] for i, el in enumerate(self.alpha_sources)])
        
    def _estimate(self, t, w_plus, z, value):
        """Temporary interface to old cvxportfolio."""
        #for cost in self.alpha_sources:
        #    cost.LEGACY = True
        return super()._estimate(t, w_plus, z, value)

    # def weight_expr(self, t, wplus, z=None, v=None):
    #     """Returns the estimated alpha.
    #
    #     Args:
    #         t: time estimate is made.
    #         wplus: An expression for holdings.
    #         tau: time of alpha being estimated.
    #
    #     Returns:
    #       An expression for the alpha.
    #     """
    #     alpha = 0
    #     for idx, source in enumerate(self.alpha_sources):
    #         alpha += source.weight_expr(t, wplus) * self.weights[idx]
    #     return alpha
    #
    # def weight_expr_ahead(self, t, tau, wplus):
    #     """Returns the estimate at time t of alpha at time tau.
    #
    #     Args:
    #       t: time estimate is made.
    #       wplus: An expression for holdings.
    #       tau: time of alpha being estimated.
    #
    #     Returns:
    #       An expression for the alpha.
    #     """
    #     alpha = 0
    #     for idx, source in enumerate(self.alpha_sources):
    #         alpha += source.weight_expr_ahead(t,
    #                                           tau, wplus) * self.weights[idx]
    #     return alpha



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
        self.deltas_errors = ParameterEstimator(tmp, non_negative=True)
        super().pre_evaluation(returns, volumes, start_time, end_time, **kwargs)


class ExponentialWindowReturnsForecastErrorRisk(ReturnsForecastErrorRisk):
    """Compute returns forecast errors with exponential window of past returns.

    Currently not implemented; we need to work out the math. It's probably
    as simple as ewm(...).std() / np.sqrt(half_life).
    """

    def __init__(self):
        raise NotImplementedError


