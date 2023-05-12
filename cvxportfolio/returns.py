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
"""This module contains classes that define return models for
portfolio optimization policies, and related objects.
"""

import cvxpy as cvx
import numpy as np
import pandas as pd


from .costs import BaseCost, CombinedCosts
from .risks import BaseRiskModel
from .estimator import DataEstimator, ParameterEstimator

__all__ = [
    "ReturnsForecast",
    "ReturnsForecastError",
]


class BaseReturnsModel(BaseCost):
    """Base class for return models.

    Use this to define any logic common to return models.
    """


# class Kelly(BaseReturnsModel):
#     r"""Maximize historical log-returns."""
#
#     def __init__(self, rolling):
#         self.rolling = rolling
#
#     def pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
#         self.past_returns = cvx.Parameter((returns.shape[1], self.rolling))
#
#     def values_in_time(self, t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs):
#         self.past_returns.value = past_returns.iloc[-self.rolling:].values.T
#
#     def compile_to_cvxpy(self, w_plus, z, v):
#         return cvx.sum(cvx.log(w_plus.T @ self.past_returns + 1)) / self.rolling
    

class ReturnsForecast(BaseReturnsModel):
    r"""Returns forecast, either provided by the user or computed from the data.
    
    :param r_hat: constant or time varying returns estimates, or, if None, fitted
        from the data as historical means
    :type r_hat: pd.Series or pd.DataFrame or None
    """
    # r"""Returns forecast, either provided by the user or computed from the data.
    #
    # This class represents the term :math:`\hat{r}_t`,
    # the forecast of assets' returns at time :math:`t`.
    # :ref:`Optimization-based policies` use this, typically as the first
    # element of their objectives.
    # See Chapters 4 and 5 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_
    # for more details.
    #
    # It can either get return forecasts from the user, for example
    # computed with some machine learning technique, or it can estimate them
    # automatically from the data.
    #
    # In the first case ``r_hat`` is specified as
    #
    # * :class:`float`, if :math:`\hat{r}_t` is the same for all times and assets
    # * :class:`pandas.Series` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` is the same for all assets but changes in time
    # * :class:`pandas.Series` indexed by assets' names, if :math:`\hat{r}_t` is constant in time and changes across assets
    # * :class:`pandas.DataFrame` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` changes across time and assets.
    #
    # The returns' forecast provided by the user must be supplied for all assets
    # including cash (unless it is constant across all assets, so, also cash).
    #
    # If instead ``r_hat`` is not speficied it defaults to None. This instructs
    # this class to compute :math:`\hat{r}_t` instead. It is
    # done, at each step of a :class:`BackTest`, by evaluating an average of the
    # past returns (*i.e.,* the real returns :math:`{r}_{t-1}, {r}_{t-2}, \ldots`,
    # where :math:`t` is the current time). This class implements three ways to do so.
    #
    # * The full average of all available past returns skipping :class:`numpy.nan` values.
    #   This is the default mode if no parameters are passed.
    # * a *rolling window*  average of recent returns. This is specified by setting
    #   the ``rolling`` parameter to a positive integer. If specified it takes precedence
    #   over ``ewm``. See the documentation of :class:`pandas.DataFrame.rolling` for
    #   details. It fails if the specified window contains :class:`numpy.nan` values.
    # * an *exponential moving window* average of past returns. This is specified
    #   by setting the ``halflife`` parameter to a positive integer.
    #   The exponential moving window is a weighted
    #   average where the most recent returns have a larger weight than the older
    #   ones. This is done by calling :class:`pandas.DataFrame.ewm` with the
    #   `halflife`` argument, which represents the number of periods over which
    #   the weight decays by a factor of 2. See the relevant ``pandas`` documentation
    #   for more details.
    #
    #
    # The only exception to the above is the forecast of cash return
    # :math:`{\left(\hat{r}_t\right)}_n`, for which the observed value
    # from last period :math:`{\left({r}_{t-1}\right)}_n`
    # is typically chosen. This is
    # the default behavior and is done by setting ``lastforcash`` to ``True``.
    # (If instead ``lastforcash`` is ``False``, the same averaging of past
    # returns is used to forecast the cash return as well.)
    #
    # :param r_hat: return forecasts supplied by the user, default is None
    # :type r_hat: float or pandas.Series or pandas.DataFrame or None
    # :param rolling: size of rolling window, it takes precendence over
    #     ``halflife`` if both are specified
    # :type rolling: int or None
    # :param halflife: half-life of exponential moving window
    # :type halflife: int or None
    # :param lastforcash: use last value to estimate cash return
    # :type lastforcash: bool
    #
    # :raises cvxportfolio.MissingValuesError: If the class accesses
    #     user-provided elements of ``r_hat`` that are :class:`numpy.nan`,
    #     or the rolling window used for averaging contains :class:`numpy.nan`.
    # :raises ValueError: If the data passed by the user is not of the right
    #     size (for example the cash returns' columns is missing).
    #
    # :Example:
    #
    # >>> import cvxportfolio as cp
    # >>> policy = cp.SinglePeriod(cp.ReturnsForecast() - \
    #     0.5 * cp.FullCovariance())
    #
    # Defines a single period optimization policy where the returns' forecasts
    # :math:`\hat{r}_t` are the full average of past returns at each point in time
    # and the risk model is the full covariance, also computed from the past returns.
    # """

    def __init__(self, r_hat=None, #rolling=None, halflife=None, 
                lastforcash=True, 
                subtractshorts=True):
        
        if not r_hat is None:
            self.r_hat = DataEstimator(r_hat)
        else:
            self.r_hat = None
            
        self.lastforcash = True
        self.subtractshorts = subtractshorts
        
        if self.subtractshorts:
            self.cash_return = cvx.Parameter(nonneg=True)
        
       
    @classmethod # we make it a classmethod so that also covariances can use it
    def update_full_mean(cls, past_returns, last_estimation, last_counts, last_time):

        if last_time is None: # full estimation
            estimation = past_returns.sum()
            counts = past_returns.count()
        else:
            assert last_time == past_returns.index[-2]
            estimation = last_estimation * last_counts + past_returns.iloc[-1].fillna(0.)
            counts = last_counts + past_returns.iloc[-1:].count()

        return estimation/counts, counts, past_returns.index[-1]
            
        
    def pre_evaluation(self, universe, backtest_times):
        
        self.r_hat_parameter = cvx.Parameter(len(universe))
        
    
    def values_in_time(self, t, past_returns, **kwargs):
        
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
        
        if self.r_hat is None:
            tmp = past_returns.mean()
            if self.lastforcash:
                tmp.iloc[-1] = past_returns.iloc[-1, -1]
            self.r_hat_parameter.value = tmp.values
        else:
            self.r_hat_parameter.value = self.r_hat.current_value
            
        if self.subtractshorts:
            self.cash_return.value = self.r_hat_parameter.value[-1]

        
    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        if self.subtractshorts:
            noncash = w_plus[:-1].T @ self.r_hat_parameter[:-1]
            cash = (w_plus[-1] - cvx.sum(cvx.neg(w_plus[:-1]))) * self.cash_return
            # print(cash)
            assert cash.is_concave()
            return noncash + cash
        else:
            return w_plus.T @ self.r_hat_parameter
        


class ReturnsForecastError(BaseRiskModel):
    """Simple return forecast error risk with values provided by the user.

    Implements the model described in pages 31-32 of the paper. You
    must pass the delta Series (if constant) or DataFrame (if time-varying)
    of the forecast errors. Or, use one of the classes below to
    compute it automatically. Multiply this object by an external multiplier
    (which can itself be a ParameterEstimator) and calibrate for the right
    amount of penalization.

    :param deltas_errors: constant per-symbol
        errors on the returns forecasts (if Series),
        or varying in time (if DataFrame), 
        or fitted from the data as the standard deviation
        of the historical mean estimator 
    :type deltas_errors: pd.DataFrame or pd.Series or None
    """

    def __init__(self, deltas=None, zeroforcash=True, # rolling=None, halflife=None
        ):
        
        if not deltas is None:
            self.deltas = DataEstimator(deltas)
        else:
            self.deltas = None
        self.zeroforcash = zeroforcash

            
    def pre_evaluation(self, universe, backtest_times):
        super().pre_evaluation(universe=universe, backtest_times=backtest_times)
        self.deltas_parameter = cvx.Parameter(len(universe), nonneg=True)


    def values_in_time(self, t, past_returns, **kwargs):
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
        if self.deltas is None:
            # if self.mode == 'full':
            tmp = (past_returns.iloc[:,:].std() / np.sqrt(past_returns.iloc[:,:].count())).values
            if self.zeroforcash:
                tmp[-1] = 0.
            self.deltas_parameter.value = tmp
        else:
            self.deltas_parameter.value = self.deltas.current_value


    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile to cvxpy expression."""
        return cvx.abs(w_plus_minus_w_bm).T @ self.deltas_parameter




