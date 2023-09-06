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

import cvxpy as cp
import numpy as np
import pandas as pd


from .costs import BaseCost, CombinedCosts
from .risks import BaseRiskModel
from .estimator import DataEstimator  # , ParameterEstimator
from .forecast import HistoricalMeanReturn, HistoricalMeanError

__all__ = [
    "ReturnsForecast",
    "ReturnsForecastError",
    "CashReturn",
]


class BaseReturnsModel(BaseCost):
    """Base class for return models.

    Use this to define any logic common to return models.
    """


class CashReturn(BaseReturnsModel):
    r"""Objective term representing cash return.

    By default, the forecast of cash return :math:`{\left(\hat{r}_t\right)}_n` 
    is the observed value from last period :math:`{\left({r}_{t-1}\right)}_n`.

    This object is included automatically in :class:`SinglePeriodOptimization`
    and :class:`MultiPeriodOptimization` policies. You can change
    this behavior by setting their ``include_cash_return`` to False. If you do
    so, you may include this cost explicitely in the objective. You need
    to do so (only) if you provide your own cash return forecast.

    :param cash_returns: if you have your forecast for the cash return, you
        should pass it here, either as a float (if constant) or as pd.Series
        with datetime index (if it changes in time). If you leave the default,
        None, the cash return forecast at time t is the observed cash return 
        at time t-1. (As is suggested in the book.)
    :type cash_returns: float or pd.Series or None
    """

    def __init__(self, cash_returns=None):
        self.cash_returns = None if cash_returns is None else DataEstimator(
            cash_returns, compile_parameter=True)

    def _pre_evaluation(self, universe, backtest_times):
        self.cash_return_parameter = cp.Parameter() if self.cash_returns is None \
            else self.cash_returns.parameter

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update cash return parameter as last cash return."""
        if self.cash_returns is None:
            self.cash_return_parameter.value = past_returns.iloc[-1, -1]

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Apply cash return to cash position."""
        return w_plus[-1] * self.cash_return_parameter


class ReturnsForecast(BaseReturnsModel):
    r"""Returns forecast for non-cash assets, provided by the user or computed from the data.

    This class represents the term :math:`\hat{r}_t`,
    the forecast of non-cash assets' returns at time :math:`t`.
    :ref:`Optimization-based policies` use this, typically as the first
    element of their objectives.
    See Chapters 4 and 5 of the `book <https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_
    for more details.

    It can either get return forecasts from the user, for example
    computed with some machine learning technique, or it can estimate them
    automatically from the data.

    In the first case ``r_hat`` is specified as

    * :class:`float`, if :math:`\hat{r}_t` is the same for all times and non-cash assets
    * :class:`pandas.Series` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` is the same for all assets but changes in time
    * :class:`pandas.Series` indexed by assets' names, if :math:`\hat{r}_t` is constant in time and changes across assets
    * :class:`pandas.DataFrame` with :class:`pandas.DatetimeIndex`, if :math:`\hat{r}_t` changes across time and assets.

    The returns' forecast provided by the user must be supplied for all assets
    excluding cash.

    If instead ``r_hat`` is not speficied it defaults to None. This instructs
    this class to compute :math:`\hat{r}_t` instead. It is
    done, at each step of a :class:`BackTest`, by evaluating the full average of the
    past returns (*i.e.,* the real returns :math:`{r}_{t-1}, {r}_{t-2}, \ldots`,
    where :math:`t` is the current time),  skipping :class:`numpy.nan` values.
    This is the default mode if no parameters are passed.

    :param r_hat: constant or time varying returns estimates, provided in the form of
        a pandas DataFrame indexed by timestamps of trading period and whose columns
        are all non-cash assets. Alternatively it can be a pandas Series indexed by the
        assets' names (so it is constant in time), a pandas Series indexed by time (so it is 
        constant across assets), or a float (constant for all times and assets). 
        If it is None, the default, the return forecasts are fitted from the data as historical means.
    :type r_hat: pd.Series or pd.DataFrame or float or None
    :param decay: decay factor used in :class:`MultiPeriodOptimization` policies.
        It is as a number in :math:`[0,1]`. At step :math:`\tau` of the MPO policy, where 
        :math:`\tau=t` is the initial one, the return predictions are multiplied by 
        :math:`\texttt{decay}^{\tau-t}`. So, ``decay`` close to zero models a `fast` signal
        while ``decay`` close to one a `slow` signal. The default value is 1.    
    :type decay: float

    :raises cvxportfolio.MissingTimesError: If the class accesses
        user-provided elements of ``r_hat`` that are :class:`numpy.nan`.

    :Example:

    >>> import cvxportfolio as cvx
    >>> policy = cvx.SinglePeriodOptimization(cvx.ReturnsForecast() - \
        0.5 * cvx.FullCovariance(), [cvx.LongOnly(), cvx.LeverageLimit(1)])
    >>> cvx.MarketSimulator(['AAPL', 'MSFT', 'GOOG']).backtest(policy).plot()

    Defines a single period optimization policy where the returns' forecasts
    :math:`\hat{r}_t` are the full average of past returns at each point in time
    and the risk model is the full covariance, also computed from the past returns.
    """

    def __init__(self, r_hat=None, decay=1.):

        if not r_hat is None:
            self.r_hat = DataEstimator(r_hat)
        else:
            self.r_hat = HistoricalMeanReturn()
        self.decay = decay

    def _pre_evaluation(self, universe, backtest_times):
        self.r_hat_parameter = cp.Parameter(len(universe)-1)

    def _values_in_time(self, t, past_returns, mpo_step=0, **kwargs):
        self.r_hat_parameter.value = self.r_hat.current_value * \
            self.decay**(mpo_step)

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Cvxpy expression acts on non-cash assets."""
        return w_plus[:-1].T @ self.r_hat_parameter


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

    def __init__(self, deltas=None):

        if not deltas is None:
            self.deltas = DataEstimator(deltas)
        else:
            self.deltas = HistoricalMeanError()

    def _pre_evaluation(self, universe, backtest_times):
        self.deltas_parameter = cp.Parameter(len(universe)-1, nonneg=True)

    def _values_in_time(self, t, past_returns, **kwargs):
        self.deltas_parameter.value = self.deltas.current_value

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile to cvxpy expression."""
        return cp.abs(w_plus_minus_w_bm[:-1]).T @ self.deltas_parameter
