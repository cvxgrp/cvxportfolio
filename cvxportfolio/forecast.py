# Copyright 2023 Enzo Busseti
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
r"""This module implements the simple forecasting models used by Cvxportfolio.

These are standard ones like historical mean, variance, and covariance. In
most cases the models implemented here are equivalent to the relevant Pandas
DataFrame methods, including (most importantly) the logic used to skip over any
``np.nan``. There are some subtle differences explained below.

Our forecasters are optimized to be evaluated sequentially in time: at each
point in time in a back-test the forecast computed at the previous time step
is updated with the most recent observation. This is in some cases (e.g.,
covariances) much more efficient than computing from scratch.

Most of our forecasters implement both a rolling window and exponential moving
average logic. These are specified by the ``rolling`` and ``half_life``
parameters respectively, which are either Pandas Timedeltas or ``np.inf``.
The latter is the default, and means that the whole past is used, with no
exponential smoothing. Note that it's possible to use both, *e.g.*,
estimate covariance matrices ignoring past returns older than 5 years and
smoothing the recent ones using an exponential kernel with half-life of 1 year.

Finally, we note that the covariance, variance and standard deviation
forecasters implement the ``kelly`` parameter, which is True by default.
This is a simple trick explained in
:paper:`section 4.2 (page 28) <section.4.2>` of the paper, simplifies the
computation and provides in general (slightly) higher performance.
For example, using the notation of the paper, the classical definition of
covariance is

.. math::

    \Sigma = \mathbf{E}(r_t - \mu)(r_t - \mu)^T,

this is what you get by setting ``kelly=False``. The default, ``kelly=True``,
gives instead

.. math::

    \Sigma^\text{kelly} = \mathbf{E}r_t r_t^T = \Sigma + \mu \mu^T,

so that the resulting Markowitz-style optimization problem corresponds to
the second order Taylor approximation of a (risk-constrained) Kelly objective,
as is explained briefly :paper:`at page 28 of the paper <section.4.2>`, or with
more detail (and hard-to-read math) in `section 6 of the Risk-Constrained Kelly
Gambling paper
<https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf#section.6>`_.

Lastly, some forecasters implement a basic caching mechanism.
This is used in two ways. First, online (e.g., in back-test): if multiple
copies of the same forecaster need access to the estimated value, as is the
case in :class:`cvxportfolio.MultiPeriodOptimization` policies, the expensive
evaluation is only done once. Then, offline, provided that the
:class:`cvxportfolio.data.MarketData` server used implements the
:meth:`cvxportfolio.data.MarketData.partial_universe_signature` method
(so that we can certify which market data the cached values are computed on).
This type of caching simply saves on disk the forecasted values, and makes it
available automatically next time the user runs a back-test on the same market
data (and same universe). This is especially useful when doing hyper-parameter
optimization, so that expensive computations like evaluating large covariance
matrices are only done once.

How to use them
~~~~~~~~~~~~~~~

These forecasters are each the default option of some Cvxportfolio optimization
term, for example :class:`HistoricalMeanReturn` is the default used by
:class:`cvxportfolio.ReturnsForecast`. In this way each is used with its
default options. If you want to change the options you can simply pass
the relevant forecaster class, instantiated with the options of your choice,
to the Cvxportfolio object. For example

.. code-block::

    import cvxportfolio as cvx
    from cvxportfolio.forecast import HistoricalMeanReturn
    import pandas as pd

    returns_forecast = cvx.ReturnsForecast(
        r_hat = HistoricalMeanReturn(
            half_life=pd.Timedelta(days=365),
            rolling=pd.Timedelta(days=365*5)))

if you want to apply exponential smoothing to the mean returns forecaster with
half-life of 1 year, and skip over all observations older than 5 years. Both
are relative to each point in time at which the policy is evaluated.
"""

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .errors import DataError, ForecastError
from .estimator import Estimator, SimulatorEstimator
from .hyperparameters import _resolve_hyperpar

logger = logging.getLogger(__name__)

def online_cache(values_in_time):
    """A simple online cache that decorates :meth:`values_in_time`.

    The instance it is used on needs to be hashable (we currently use
    the hash of its __repr__ via dataclass).

    :param values_in_time: :meth:`values_in_time` method to decorate.
    :type values_in_time: function

    :returns: Decorated method, which saves and retrieves (if available) from
        cache the result.
    :rtype: function
    """

    def wrapped(self, t, cache=None, **kwargs):
        """Cached :meth:`values_in_time` method.

        :param self: Class instance.
        :type self: cvxportfolio.Estimator
        :param t: Current time, used as in the key for caching.
        :type t: pandas.Timestamp
        :param cache: Cache dictionary; if None (the default) caching is
            disabled.
        :type cache: dict or None
        :param kwargs: Extra arguments that are passed through.
        :type kwargs: dict

        :returns: The returned value, maybe retrieved from cache.
        :rtype: float or numpy.array
        """

        if cache is None:  # temporary to not change tests
            cache = {}

        if not self in cache:
            cache[self] = {}

        if t in cache[self]:
            logger.debug(
                '%s.values_in_time at time %s is retrieved from cache.',
                self, t)
            result = cache[self][t]
        else:
            result = values_in_time(self, t=t, cache=cache, **kwargs)
            logger.debug('%s.values_in_time at time %s is stored in cache.',
                self, t)
            cache[self][t] = result
        return result

    return wrapped


class BaseForecast(Estimator):
    """Base class for forecasters."""

    _last_time = None

    def __post_init__(self):
        raise NotImplementedError # pragma: no cover

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.__post_init__()

    def _agnostic_update(self, t, past_returns, **kwargs):
        """Choose whether to make forecast from scratch or update last one."""
        if (self._last_time is None) or (
            self._last_time != past_returns.index[-1]):
            logger.debug(
                '%s.values_in_time at time %s is computed from scratch.',
                self, t)
            self._initial_compute(t=t, past_returns=past_returns, **kwargs)
        else:
            logger.debug(
              '%s.values_in_time at time %s is updated from previous value.',
              self, t)
            self._online_update(t=t, past_returns=past_returns, **kwargs)

    def _initial_compute(self, **kwargs):
        """Make forecast from scratch."""
        raise NotImplementedError # pragma: no cover

    def _online_update(self, **kwargs):
        """Update forecast from period before."""
        raise NotImplementedError # pragma: no cover

def _is_timedelta(value):
    if isinstance(value, pd.Timedelta):
        if value <= pd.Timedelta('0d'):
            raise ValueError(
                '(Exponential) moving average window must be positive')
        return True
    if isinstance(value, float) and np.isposinf(value):
        return False
    raise ValueError(
        '(Exponential) moving average window can only be'
        ' pandas Timedeltas or np.inf.')


@dataclass(unsafe_hash=True)
class BaseMeanVarForecast(BaseForecast):
    """This class contains logic common to mean and (co)variance forecasters.

    It implements both moving average and exponential moving average, which
    can be used at the same time (e.g., ignore observations older than 5
    years and weight exponentially with half-life of 1 year the recent ones).

    Then, it implements the "online update" vs "compute from scratch" model,
    updating with a new observations is much cheaper than computing from
    scratch (especially for covariances).
    """

    half_life: Union[pd.Timedelta, float] = np.inf
    rolling: Union[pd.Timedelta, float] = np.inf

    def __post_init__(self):
        self._last_time = None
        self._denominator = None
        self._numerator = None

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        raise NotImplementedError # pragma: no cover

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        raise NotImplementedError # pragma: no cover

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _dataframe_selector(self, **kwargs):
        """Return dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        raise NotImplementedError # pragma: no cover

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Obtain current value of the historical mean of given dataframe.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Historical means of given dataframe.
        :rtype: numpy.array
        """
        self._agnostic_update(**kwargs)
        return (self._numerator / self._denominator).values

    def _emw_weights(self, index, t):
        """Get weights to apply to the past obs for EMW."""
        index_in_halflifes = (index - t) / _resolve_hyperpar(self.half_life)
        return np.exp(index_in_halflifes * np.log(2))

    def _initial_compute(self, t, **kwargs): # pylint: disable=arguments-differ
        """Make forecast from scratch.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector(t=t, **kwargs)

        # Moving average window logic
        if _is_timedelta(_resolve_hyperpar(self.rolling)):
            df = df.loc[df.index >= t-_resolve_hyperpar(self.rolling)]

        # If EMW, compute weights here
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(df.index, t)
        else:
            emw_weights = None

        self._denominator = self._compute_denominator(df, emw_weights)
        if np.min(self._denominator.values) == 0:
            raise ForecastError(
                f'{self.__class__.__name__} is given a dataframe with '
                    + 'at least a column that has no values.')
        self._numerator = self._compute_numerator(df, emw_weights)
        self._last_time = t

        # used by covariance forecaster
        return df, emw_weights

    def _online_update(self, t, **kwargs): # pylint: disable=arguments-differ
        """Update forecast from period before.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector(t=t, **kwargs)
        last_row = df.iloc[-1]

        # if emw discount past
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            time_passed_in_halflifes = (
                self._last_time - t)/_resolve_hyperpar(self.half_life)
            discount_factor = np.exp(time_passed_in_halflifes * np.log(2))
            self._denominator *= discount_factor
            self._numerator *= discount_factor
        else:
            discount_factor = 1.

        # for emw we also need to discount last element
        self._denominator += self._update_denominator(
            last_row) * discount_factor
        self._numerator += self._update_numerator(last_row) * discount_factor

        # Moving average window logic: subtract elements that have gone out
        if _is_timedelta(_resolve_hyperpar(self.rolling)):
            observations_to_subtract, emw_weights_of_subtract = \
                self._remove_part_gone_out_of_ma(df, t)
        else:
            observations_to_subtract, emw_weights_of_subtract = None, None

        self._last_time = t

        # used by covariance forecaster
        return (
            discount_factor, observations_to_subtract, emw_weights_of_subtract)

    def _remove_part_gone_out_of_ma(self, df, t):
        """Subtract from numerator and denominator too old observations."""

        observations_to_subtract = df.loc[
            (df.index >= (self._last_time - _resolve_hyperpar(self.rolling)))
            & (df.index < (t - _resolve_hyperpar(self.rolling)))]

        # If EMW, compute weights here
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(observations_to_subtract.index, t)
        else:
            emw_weights = None

        self._denominator -= self._compute_denominator(
            observations_to_subtract, emw_weights)
        if np.min(self._denominator.values) == 0:
            raise ForecastError(
                f'{self.__class__.__name__} is given a dataframe with '
                + 'at least a column that has no values.')
        self._numerator -= self._compute_numerator(
            observations_to_subtract, emw_weights).fillna(0.)

        # used by covariance forecaster
        return observations_to_subtract, emw_weights


@dataclass(unsafe_hash=True)
class BaseMeanForecast(BaseMeanVarForecast): # pylint: disable=abstract-method
    """This class contains the logic common to the mean forecasters."""

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        if emw_weights is None:
            return df.sum()
        return df.multiply(emw_weights, axis=0).sum()

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        if emw_weights is None:
            return df.count()
        ones = (~df.isnull()) * 1.
        return ones.multiply(emw_weights, axis=0).sum()

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        return last_row.fillna(0.)

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        return ~(last_row.isnull())


@dataclass(unsafe_hash=True)
class HistoricalMeanReturn(BaseMeanForecast):
    r"""Historical means of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].mean()

    where ``past_returns`` is a time-indexed dataframe containing the past
    returns (if in back-test that's relative to each point in time, ), and its
    last column, which we skip over, are the cash returns. We use the same
    logic as Pandas to handle ``np.nan`` values.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical means of."""
        return past_returns.iloc[:, :-1]

@dataclass(unsafe_hash=True)
class HistoricalMeanVolume(BaseMeanForecast):
    r"""Historical means of traded volume in units of value (e.g., dollars).

    .. versionadded:: 1.2.0

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_volumes, **kwargs):
        """Return dataframe to compute the historical means of."""
        if past_volumes is None:
            raise DataError(
                f"{self.__class__.__name__} can only be used if MarketData"
                + " provides market volumes.")
        return past_volumes

@dataclass(unsafe_hash=True)
class HistoricalVariance(BaseMeanForecast):
    r"""Historical variances of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].var(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        (past_returns**2).iloc[:,:-1].mean()

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` compute :math:`\mathbf{E}[r^2]`, else
        :math:`\mathbf{E}[r^2] - {\mathbf{E}[r]}^2`. The second corresponds
        to the classic definition of variance, while the first is what is
        obtained by Taylor approximation of the Kelly gambling objective.
        See discussion above.
    :type kelly: bool
    """

    kelly: bool = True

    def __post_init__(self):
        if not self.kelly:
            self.meanforecaster = HistoricalMeanReturn(
                half_life=_resolve_hyperpar(self.half_life),
                rolling=_resolve_hyperpar(self.rolling))
        super().__post_init__()

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Variances of past returns (excluding cash).
        :rtype: numpy.array
        """
        result = super().values_in_time(**kwargs)
        if not self.kelly:
            result -= self.meanforecaster.current_value**2
        return result

    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical means of."""
        return past_returns.iloc[:, :-1]**2


@dataclass(unsafe_hash=True)
class HistoricalStandardDeviation(HistoricalVariance, SimulatorEstimator):
    """Historical standard deviation of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].std(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        np.sqrt((past_returns**2).iloc[:,:-1].mean())

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default True.
    :type kelly: bool
    """

    kelly: bool = True

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviations of past returns (excluding cash).
        :rtype: numpy.array
        """
        variances = \
            super().values_in_time(**kwargs)
        return np.sqrt(variances)

    def simulate(self, **kwargs):
        # TODO could take last return as well
        return self.values_in_time(
            t=kwargs['t'],
            # These are not necessary with current design of
            # DataEstimator
            current_weights=kwargs['current_weights'],
            current_portfolio_value=kwargs['current_portfolio_value'],
            past_returns=kwargs['past_returns'],
            past_volumes=kwargs['past_volumes'],
            current_prices=kwargs['current_prices']
        )

@dataclass(unsafe_hash=True)
class HistoricalMeanError(HistoricalVariance):
    r"""Historical standard deviations of the mean of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2},
    \ldots, r_0` this is :math:`\sqrt{\text{Var}[r]/t}`. When there are
    missing values we ignore them, both to compute the variance and the
    count.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default False.
    :type kelly: bool
    """

    kelly: bool = False

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviation of the mean of past returns (excluding
            cash).
        :rtype: numpy.array
        """
        variance = super().values_in_time(**kwargs)
        return np.sqrt(variance / self._denominator.values)


@dataclass(unsafe_hash=True)
class HistoricalCovariance(BaseMeanVarForecast):
    r"""Historical covariance matrix."""

    kelly: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._joint_mean = None

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        filled = df.fillna(0.)
        if emw_weights is None:
            return filled.T @ filled
        tmp = filled.multiply(emw_weights, axis=0)
        return tmp.T @ filled

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        ones = (~df.isnull()) * 1.
        if emw_weights is None:
            return ones.T @ ones
        tmp = ones.multiply(emw_weights, axis=0)
        return tmp.T @ ones

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        nonnull = ~(last_row.isnull())
        return np.outer(nonnull, nonnull)

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        filled = last_row.fillna(0.)
        return np.outer(filled, filled)

    def _dataframe_selector( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Return dataframe to compute the historical covariance of."""
        return past_returns.iloc[:, :-1]

    def _compute_joint_mean(self, df, emw_weights):
        r"""Compute precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        nonnull = (~df.isnull()) * 1.
        if emw_weights is None:
            return nonnull.T @ df.fillna(0.)
        return nonnull.T @ df.fillna(0.).multiply(emw_weights, axis=0)

    def _update_joint_mean(self, last_row):
        r"""Update precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        return last_row.fillna(0.)

    def _initial_compute( # pylint: disable=arguments-differ
            self, **kwargs):
        """Compute from scratch, taking care of non-Kelly correction."""

        df, emw_weights = super()._initial_compute(**kwargs)

        if not self.kelly:
            self._joint_mean = self._compute_joint_mean(df, emw_weights)

    def _online_update( # pylint: disable=arguments-differ
            self, **kwargs):
        """Update from last observation."""

        discount_factor, observations_to_subtract, emw_weights_of_subtract = \
            super()._online_update(**kwargs)
        df = self._dataframe_selector(**kwargs)
        last_row = df.iloc[-1]

        if not self.kelly:

            # discount past if EMW
            if discount_factor != 1.:
                self._joint_mean *= discount_factor

            # add last anyways
            self._joint_mean += self._update_joint_mean(
                last_row) * discount_factor

            # if MW, update by removing old observations
            if observations_to_subtract is not None:
                self._joint_mean -= self._compute_joint_mean(
                    observations_to_subtract, emw_weights_of_subtract)

    def values_in_time(self, **kwargs):
        """Obtain current value of the covariance estimate.

        :param kwargs: All arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        covariance = super().values_in_time(**kwargs)

        if not self.kelly:
            tmp = self._joint_mean / self._denominator
            covariance -= tmp.T * tmp

        return covariance

def project_on_psd_cone_and_factorize(covariance):
    """Factorize matrix and remove negative eigenvalues.

    :param covariance: Square (symmetric) approximate covariance matrix, can
         have negative eigenvalues.
    :type covariance: numpy.array

    :returns: Square root factorization with negative eigenvalues removed.
    :rtype: numpy.array
    """
    eigval, eigvec = np.linalg.eigh(covariance)
    eigval = np.maximum(eigval, 0.)
    return eigvec @ np.diag(np.sqrt(eigval))

@dataclass(unsafe_hash=True)
class HistoricalFactorizedCovariance(HistoricalCovariance):
    r"""Historical covariance matrix of non-cash returns, factorized.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent
    to, before factorization

    .. code-block::

        past_returns.iloc[:,:-1].cov(ddof=0)

    if you set ``kelly=False``. We use the same logic to handle ``np.nan``
    values. For ``kelly=True`` it is not possible to reproduce with one single
    Pandas method (but we do test against Pandas in the unit tests).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` each element of the covariance matrix
        :math:`\Sigma_{i,j}` is equal to :math:`\mathbf{E} r^{i} r^{j}`,
        otherwise it is
        :math:`\mathbf{E} r^{i} r^{j} - \mathbf{E} r^{i} \mathbf{E} r^{j}`.
        The second case corresponds to the classic definition of covariance,
        while the first is what is obtained by Taylor approximation of
        the Kelly gambling objective. (See discussion above.)
        In the second case, the estimated covariance is the same
        as what is returned by ``pandas.DataFrame.cov(ddof=0)``, *i.e.*,
        we use the same logic to handle missing data.
    :type kelly: bool
    """

    # this is used by FullCovariance
    FACTORIZED = True

    @online_cache
    def values_in_time( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Obtain current value of the covariance estimate.

        :param t: Current time period (possibly of simulation).
        :type t: pandas.Timestamp
        :param kwargs: All arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.ForecastError: The procedure failed,
            typically because there are too many missing values (*e.g.*, some
            asset has only missing values).

        :returns: Square root factorized covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        covariance = super().values_in_time(t=t, **kwargs)

        try:
            return project_on_psd_cone_and_factorize(covariance)
        except np.linalg.LinAlgError as exc: # pragma: no cover
            raise ForecastError(f'Covariance estimation at time {t} failed;'
                + ' there are (probably) too many missing values in the'
                + ' past returns.') from exc


@dataclass(unsafe_hash=True)
class HistoricalLowRankCovarianceSVD(Estimator):
    """Build factor model covariance using truncated SVD.

    .. note::

        This forecaster is experimental and not covered by semantic versioning,
        we may change it without warning.

    :param num_factors: How many factors in the low rank model.
    :type num_factors: int
    :param svd_iters: How many iteration of truncated SVD to apply. If you
        get a badly conditioned covariance you may to lower this.
    :type svd_iters: int
    :param svd: Which SVD routine to use, currently only dense (LAPACK) via
        Numpy.
    :type svd: str
    """

    num_factors: int
    svd_iters: int = 10
    svd: str = 'numpy'

    # brought back from old commit;
    #
    #  https://github.com/cvxgrp/cvxportfolio/commit
    #     /aa3d2150d12d85a6fb1befdf22cb7967fcc27f30
    #
    # matches original 2016 method from example
    # notebooks with new heuristic for NaNs. can probably be improved
    # by terminating early if idyosyncratic becomes negative

    @staticmethod
    def build_low_rank_model(rets, num_factors=10, iters=10, svd='numpy'):
        r"""Build a low rank risk model from past returns that include NaNs.

        This is an experimental procedure that may work well on past
        returns matrices with few NaN values (say, below 20% of the
        total entries). If there are (many) NaNs, one should probably
        also use a rather large risk forecast error.

        :param rets: Past returns, excluding cash.
        :type rets: pandas.DataFrame
        :param num_factors: How many factors in the fitted model.
        :type num_factors: int
        :param iters: How many iterations of SVD are performed.
        :type iters: int
        :param svd: Which singular value decomposition routine is used,
            default (and currently the only one supported) is ``'numpy'``.
        :type svd: str

        :raises SyntaxError: If wrong ``svd`` parameter is supplied.
        :raises cvxportfolio.errors.ForecastError: If the procedure fails;
             you may try with lower ``num_factors`` or ``iters``.

        :returns: (F, d)
        :rtype: (numpy.array, numpy.array)
        """
        nan_fraction = rets.isnull().sum().sum() / np.prod(rets.shape)
        normalizer = np.sqrt((rets**2).mean())
        normalized = rets
        if nan_fraction:
            nan_implicit_imputation = pd.DataFrame(0.,
                columns=normalized.columns, index = normalized.index)
            for _ in range(iters):
                if svd == 'numpy':
                    u, s, v = np.linalg.svd(
                        normalized.fillna(nan_implicit_imputation),
                        full_matrices=False)
                else:
                    raise SyntaxError(
                        'Currently only numpy svd is implemented')
                nan_implicit_imputation = pd.DataFrame(
                    (u[:, :num_factors] * (s[:num_factors]
                        )) @ v[:num_factors],
                    columns = normalized.columns, index = normalized.index)
        else:
            if svd == 'numpy':
                u, s, v = np.linalg.svd(normalized, full_matrices=False)
            else:
                raise SyntaxError(
                    'Currently only numpy svd is implemented')
        F = v[:num_factors].T * s[:num_factors] / np.sqrt(len(rets))
        F = pd.DataFrame(F.T, columns=normalizer.index)
        idyosyncratic = normalizer**2 - (F**2).sum(0)
        if not np.all(idyosyncratic >= 0.):
            raise ForecastError(
                "Low rank risk estimation with iterative SVD did not work."
                + " You probably have too many missing values in the past"
                + " returns. You may try with HistoricalFactorizedCovariance,"
                + " or change your universe.")
        return F.values, idyosyncratic.values

    @online_cache
    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Current low-rank model, also cached.

        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Extra arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Low-rank plus diagonal covariance model: (F, d); excludes
             cash.
        :rtype: (numpy.array, numpy.array)
        """
        return self.build_low_rank_model(past_returns.iloc[:, :-1],
            num_factors=self.num_factors,
            iters=self.svd_iters, svd=self.svd)
