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
"""This module contains classes that define return models for portfolio
optimization policies and related objects."""

import cvxpy as cp

from .costs import Cost
from .estimator import DataEstimator  # , ParameterEstimator
from .forecast import HistoricalMeanError, HistoricalMeanReturn

__all__ = [
    "ReturnsForecast",
    "ReturnsForecastError",
    "CashReturn",
]


class CashReturn(Cost):
    r"""Objective term representing cash return.

    By default, the forecast of cash return :math:`{\left(\hat{r}_t\right)}_n`
    is the observed value from last period :math:`{\left({r}_{t-1}\right)}_n`.

    This object is included automatically in
    :class:`cvxportfolio.SinglePeriodOptimization` and
    :class:`cvxportfolio.MultiPeriodOptimization` policies. You can change
    this behavior by setting their ``include_cash_return`` argument to False.
    If you do so, you may include this cost explicitely in the objective. You
    need to do so (only) if you provide your own cash return forecast.

    :param cash_returns: if you have your forecast for the cash return, you
        should pass it here, either as a float (if constant) or as pd.Series
        with datetime index (if it changes in time). If you leave the default,
        None, the cash return forecast at time t is the observed cash return
        at time t-1.
    :type cash_returns: float or pd.Series or None
    """

    def __init__(self, cash_returns=None):
        self.cash_returns = None if cash_returns is None else DataEstimator(
            cash_returns, compile_parameter=True)
        self._cash_return_parameter = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Initialize model.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._cash_return_parameter = (cp.Parameter()
            if self.cash_returns is None else self.cash_returns.parameter)

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Update cash return parameter as last cash return.

        :param past_returns: Past market returns.
        :type past_returns: pandas.DataFrame
        :param kwargs: All other parameters to :meth:`values_in_time`.
        :type kwargs: dict
        """
        if self.cash_returns is None:
            self._cash_return_parameter.value = past_returns.iloc[-1, -1]

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression representing the risk model.
        :rtype: cvxpy.expression
        """
        return w_plus[-1] * self._cash_return_parameter


class ReturnsForecast(Cost):
    r"""Returns forecast for non-cash assets, provided by the user or computed
    from the data.

    It represents the objective term:

    .. math::

        \hat{r}_t^T z_t.

    :ref:`Optimization-based policies` use this, typically as the first
    element of their objectives. See chapters 4 and 5 of the paper, for example
    :paper:`page 25 <section.4.1>` for more details.

    This class can either get return forecasts from the user, for example
    computed with some machine learning technique, or it can estimate them
    automatically from the data. The returns' forecast provided by the user 
    must be supplied for all assets excluding cash. See the :ref:`passing-data`
    manual page for more information on how these are passed.

    :param r_hat: constant or time varying returns estimates, provided in the
        form of a pandas DataFrame indexed by timestamps of trading period and
        whose columns are all non-cash assets. Alternatively it can be a pandas
        Series indexed by the assets' names (so it is constant in time), a
        pandas Series indexed by time (so it is constant across assets), or a 
        float (constant for all times and assets). Alternatively you can
        provide a :class:`cvxportfolio.estimator.Estimator` subclass that 
        implements the logic to compute the returns forecast given the past
        market data, like the default 
        :class:`cvxportfolio.forecast.HistoricalMeanReturn` which computes the
        historical means of the past returns, at each point in the back-test.
    :type r_hat: pd.Series or pd.DataFrame or float or 
        :class:`cvxportfolio.estimator.Estimator`
    :param decay: decay factor used in 
        :class:`cvxportfolio.MultiPeriodOptimization` policies. It is as a 
        number in :math:`[0,1]`. At step :math:`\tau` of the MPO policy, where 
        :math:`\tau=0` is the initial one, the return predictions are
        multiplied by :math:`\texttt{decay}^{\tau}`. So, ``decay`` close to
        zero models a `fast` signal while ``decay`` close to one a `slow`
        signal. The default value is 1.    
    :type decay: float

    :Example:

    >>> import cvxportfolio as cvx
    >>> policy = cvx.SinglePeriodOptimization(cvx.ReturnsForecast() - \
        0.5 * cvx.FullCovariance(), [cvx.LongOnly(), cvx.LeverageLimit(1)])
    >>> cvx.MarketSimulator(['AAPL', 'MSFT', 'GOOG']).backtest(policy).plot()

    Defines a single period optimization policy where the returns' forecasts
    :math:`\hat{r}_t` are the full average of past returns at each point in 
    time and the risk model is the full covariance, also computed from the past 
    returns.
    """

    def __init__(self, r_hat=HistoricalMeanReturn, decay=1.):

        if isinstance(r_hat, type):
            r_hat = r_hat()

        # we don't use DataEstimator's parameter
        # because we apply the decay
        self.r_hat = DataEstimator(r_hat)
        self.decay = decay
        self._r_hat_parameter = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize model with universe size.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._r_hat_parameter = cp.Parameter(len(universe)-1)

    def values_in_time( # pylint: disable=arguments-differ
            self, mpo_step=0, **kwargs):
        """Update returns parameter knowing which MPO step we're at.

        :param mpo_step: MPO step, 0 is current.
        :type mpo_step: int
        :param kwargs: All other parameters to :meth:`values_in_time`.
        :type kwargs: dict
        """
        self._r_hat_parameter.value = self.r_hat.current_value *\
            self.decay**(mpo_step)

    def compile_to_cvxpy(self,  w_plus, z, w_plus_minus_w_bm):
        """Compile to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression representing the risk model.
        :rtype: cvxpy.expression
        """
        return w_plus[:-1].T @ self._r_hat_parameter


class ReturnsForecastError(Cost):
    r"""Simple return forecast error risk with values provided by the user.

    It represents the objective term:

    .. math::

        \delta^T |w^+_t - w^\text{b}_t |

    Implements the model described in :paper:`chapter 4, page 31 <section.4.3>`
    of the paper. You can pass the penalization parameters (see
    the :ref:`passing-data` manual page) or rely on a forecaster to do so.
    The default is :class:`cvxportfolio.forecast.HistoricalMeanError`, which
    computes the standard deviation of the mean for each asset's past returns,
    iteratively during a back-test.

    :param deltas_errors: Constant per-symbol errors on the returns
        forecasts (if Series), or varying in time (if DataFrame), or
        fitted from the data as the standard deviation of the historical
        mean estimator.
    :type deltas_errors: pd.DataFrame or pd.Series
        or :class:`cvxportfolio.estimator.Estimator`:
    """

    def __init__(self, deltas=HistoricalMeanError):

        if isinstance(deltas, type):
            deltas = deltas()
        self.deltas = DataEstimator(deltas)
        self._deltas_parameter = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize model with universe size.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: pandas.DatetimeIndex
        """
        self._deltas_parameter = cp.Parameter(len(universe)-1, nonneg=True)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Update returns forecast error parameters.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """
        self._deltas_parameter.value = self.deltas.current_value

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression representing the risk model.
        :rtype: cvxpy.expression
        """
        return cp.abs(w_plus_minus_w_bm[:-1]).T @ self._deltas_parameter
