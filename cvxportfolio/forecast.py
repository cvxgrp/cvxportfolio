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
"""This module contains classes that make forecasts.

It implements the models described in Chapter 7 of the book (Examples).

For example, historical means of market returns, and covariances, are
forecasted here. These are used internally by cvxportfolio objects. In
addition, some of the classes defined here have the ability to cache the
result of their computation online so that if multiple copies of the
same forecaster need access to the estimated value (as is the case in
MultiPeriodOptimization policies) the expensive evaluation is only done
once. The same cache is stored on disk when a back-test ends, so next
time the user runs a back-test with the same universe and market data,
the forecasted values will be retrieved automatically.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import ForecastError
from .estimator import Estimator

logger = logging.getLogger(__name__)

def online_cache(values_in_time):
    """A simple online cache that decorates values_in_time.

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
    and updating with a new observations is much cheaper than computing from
    scratch (especially for covariances).
    """

    ema_half_life: pd.Timedelta = np.inf
    ma_window: pd.Timedelta = np.inf

    def __post_init__(self):
        self._last_time = None
        self._denominator = None
        self._numerator = None

    def _compute_numerator(self, df, ewm_weights):
        """Exponential moving window (optional) numerator."""
        raise NotImplementedError # pragma: no cover

    def _compute_denominator(self, df, ewm_weights):
        """Exponential moving window (optional) denominator."""
        raise NotImplementedError # pragma: no cover

    def _update_numerator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _update_denominator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _dataframe_selector(self, **kwargs):
        """Return dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        .
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

    def _ewm_weights(self, index, t):
        index_in_halflifes = (index - t) / self.ema_half_life
        return np.exp(index_in_halflifes * np.log(2))

    def _initial_compute(self, t, **kwargs): # pylint: disable=arguments-differ
        """Make forecast from scratch.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector(t=t, **kwargs)

        # Moving average window logic
        if _is_timedelta(self.ma_window):
            df = df.loc[df.index >= t-self.ma_window]

        # If EWM, compute weights here
        if _is_timedelta(self.ema_half_life):
            ewm_weights = self._ewm_weights(df.index, t)
        else:
            ewm_weights = None

        self._denominator = self._compute_denominator(df, ewm_weights)
        if np.min(self._denominator.values) == 0:
            raise ForecastError(
                f'{self.__class__.__name__} is given a dataframe with '
                    + 'at least a column that has no values.')
        self._numerator = self._compute_numerator(df, ewm_weights)
        self._last_time = t

        # used by covariance forecaster
        return ewm_weights

    def _online_update(self, t, **kwargs): # pylint: disable=arguments-differ
        """Update forecast from period before.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector(t=t, **kwargs)
        last_row = df.iloc[-1]

        # if ewm discount past
        if _is_timedelta(self.ema_half_life):
            time_passed_in_halflifes = (
                self._last_time - t)/self.ema_half_life
            discount_factor = np.exp(time_passed_in_halflifes * np.log(2))
            self._denominator *= discount_factor
            self._numerator *= discount_factor
        else:
            discount_factor = 1.

        # for ewm we also need to discount last element
        self._denominator += self._update_denominator(
            last_row) * discount_factor
        self._numerator += self._update_numerator(last_row) * discount_factor

        # Moving average window logic: subtract elements that have gone out
        if _is_timedelta(self.ma_window):
            observations_to_subtract, ewm_weights_of_subtract = \
                self._remove_part_gone_out_of_ma(df, t)
        else:
            observations_to_subtract, ewm_weights_of_subtract = None, None

        self._last_time = t

        # used by covariance forecaster
        return (
            discount_factor, observations_to_subtract, ewm_weights_of_subtract)

    def _remove_part_gone_out_of_ma(self, df, t):
        """Subtract from numerator and denominator the observations not in MW."""

        observations_to_subtract = df.loc[
            (df.index >= (self._last_time - self.ma_window))
            & (df.index < (t - self.ma_window))]

        # If EWM, compute weights here
        if _is_timedelta(self.ema_half_life):
            ewm_weights = self._ewm_weights(observations_to_subtract.index, t)
        else:
            ewm_weights = None

        self._denominator -= self._compute_denominator(
            observations_to_subtract, ewm_weights)
        if np.min(self._denominator.values) == 0:
            raise ForecastError(
                f'{self.__class__.__name__} is given a dataframe with '
                + 'at least a column that has no values.')
        self._numerator -= self._compute_numerator(
            observations_to_subtract, ewm_weights).fillna(0.)

        # used by covariance forecaster
        return observations_to_subtract, ewm_weights


@dataclass(unsafe_hash=True)
class BaseMeanForecast(BaseMeanVarForecast): # pylint: disable=abstract-method
    """This class contains the logic common to the mean forecasters."""

    def _compute_numerator(self, df, ewm_weights):
        """Exponential moving window (optional) numerator."""
        if ewm_weights is None:
            return df.sum()
        return df.multiply(ewm_weights, axis=0).sum()

    def _compute_denominator(self, df, ewm_weights):
        """Exponential moving window (optional) denominator."""
        if ewm_weights is None:
            return df.count()
        ones = (~df.isnull()) * 1.
        return ones.multiply(ewm_weights, axis=0).sum()

    def _update_numerator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied upstream.
        """
        return last_row.fillna(0.)

    def _update_denominator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied upstream.
        """
        return ~(last_row.isnull())


@dataclass(unsafe_hash=True)
class HistoricalMeanReturn(BaseMeanForecast):
    r"""Historical mean returns.

    This ignores both the cash returns column and all missing values.
    """
    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical means of."""
        return past_returns.iloc[:, :-1]


@dataclass(unsafe_hash=True)
class HistoricalVariance(BaseMeanForecast):
    r"""Historical variances of non-cash returns.

    :param kelly: if ``True`` compute :math:`\mathbf{E}[r^2]`, else
        :math:`\mathbf{E}[r^2] - {\mathbf{E}[r]}^2`. The second corresponds
        to the classic definition of variance, while the first is what is
        obtained by Taylor approximation of the Kelly gambling objective.
        (See page 28 of the book.)
    :type kelly: bool
    """

    kelly: bool = True

    def __post_init__(self):
        if not self.kelly:
            self.meanforecaster = HistoricalMeanReturn(
                ema_half_life=self.ema_half_life,
                ma_window=self.ma_window)
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
class HistoricalStandardDeviation(HistoricalVariance):
    """Historical standard deviation."""

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

@dataclass(unsafe_hash=True)
class HistoricalMeanError(HistoricalVariance):
    r"""Historical standard deviations of the mean of non-cash returns.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2},
    \ldots, r_0` this is :math:`\sqrt{\text{Var}[r]/t}`. When there are
    missing values we ignore them, both to compute the variance and the
    count.
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

    def _compute_numerator(self, df, ewm_weights):
        """Exponential moving window (optional) numerator."""
        filled = df.fillna(0.)
        if ewm_weights is None:
            return filled.T @ filled
        tmp = filled.multiply(ewm_weights, axis=0)
        return tmp.T @ filled

    def _compute_denominator(self, df, ewm_weights):
        """Exponential moving window (optional) denominator."""
        ones = (~df.isnull()) * 1.
        if ewm_weights is None:
            return ones.T @ ones
        tmp = ones.multiply(ewm_weights, axis=0)
        return tmp.T @ ones

    def _update_denominator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied upstream.
        """
        nonnull = ~(last_row.isnull())
        return np.outer(nonnull, nonnull)

    def _update_numerator(self, last_row):
        """Update with last observation.

        Ewm (if any) is applied upstream.
        """
        filled = last_row.fillna(0.)
        return np.outer(filled, filled)

    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical covariance of."""
        return past_returns.iloc[:, :-1]

    # @staticmethod
    # def _get_count_matrix(past_returns): # -> _ewn_denominator
    #     r"""We obtain the matrix of non-null joint counts:

    #     .. math::

    #         \text{Count}\left(r^{i}r^{j} \neq \texttt{nan}\right).
    #     """
    #     df = past_returns.iloc[:, :-1]
    #     return HistoricalCovariance._compute_denominator(None, df, None)

    #     # tmp = (~past_returns.iloc[:, :-1].isnull()) * 1.
    #     # return tmp.T @ tmp

    def _compute_joint_mean(self, df, ewm_weights):
        r"""Compute precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        # TODO: ewm
        nonnull = (~df.isnull()) * 1.
        tmp = nonnull.T @ df.fillna(0.)
        return tmp  # * tmp.T

    def _update_joint_mean(self, last_row):
        r"""Update precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        return last_row.fillna(0.)

    # pylint: disable=arguments-differ
    def _initial_compute(self, **kwargs):

        ewm_weights = super()._initial_compute(**kwargs)
        df = self._dataframe_selector(**kwargs)

        #df = past_returns.iloc[:, :-1]
        #self._denominator = self._compute_denominator(df, None)
        #self._numerator = self._compute_numerator(df, None)

        if not self.kelly:
            self._joint_mean = self._compute_joint_mean(df, ewm_weights)

        #self._last_time = kwargs['t']

    def _online_update(self, **kwargs):
        """Update from last observation."""

        discount_factor, observations_to_subtract, ewm_weights_of_subtract = \
            super()._online_update(**kwargs)
        df = self._dataframe_selector(**kwargs)
        last_row = df.iloc[-1]

        # last_row = past_returns.iloc[-1, :-1]
        # self._denominator += self._update_denominator(last_row)
        # self._numerator += self._update_numerator(last_row)
        # # last_ret = past_returns.iloc[-1, :-1].fillna(0.)
        # # self._numerator += np.outer(last_ret, last_ret)
        # self._last_time = t
        if not self.kelly:
            self._joint_mean += self._update_joint_mean(
                last_row) * discount_factor

            # MA update
            if observations_to_subtract is not None:
                self._joint_mean -= self._compute_joint_mean(
                    observations_to_subtract, ewm_weights_of_subtract)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
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
    r"""Historical covariance matrix, sqrt factorized.

    :param kelly: if ``True`` compute each
        :math:`\Sigma_{i,j} = \overline{r^{i} r^{j}}`, else
        :math:`\overline{r^{i} r^{j}} - \overline{r^{i}}\overline{r^{j}}`.
        The second case corresponds to the classic definition of covariance,
        while the first is what is obtained by Taylor approximation of
        the Kelly gambling objective. (See page 28 of the book.)
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
        except np.linalg.LinAlgError as exc:
            raise ForecastError(f'Covariance estimation at time {t} failed;'
                + ' there are (probably) too many missing values in the'
                + ' past returns.') from exc


@dataclass(unsafe_hash=True)
class HistoricalLowRankCovarianceSVD(Estimator):
    """Build factor model covariance using truncated SVD."""

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
