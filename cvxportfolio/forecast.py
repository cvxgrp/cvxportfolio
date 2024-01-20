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

    def _agnostic_update(self, t, past_returns):
        """Choose whether to make forecast from scratch or update last one."""
        if (self._last_time is None) or (
            self._last_time != past_returns.index[-1]):
            logger.debug(
                '%s.values_in_time at time %s is computed from scratch.',
                self, t)
            self._initial_compute(t=t, past_returns=past_returns)
        else:
            logger.debug(
              '%s.values_in_time at time %s is updated from previous value.',
              self, t)
            self._online_update(t=t, past_returns=past_returns)

    def _initial_compute(self, t, past_returns):
        """Make forecast from scratch."""
        raise NotImplementedError # pragma: no cover

    def _online_update(self, t, past_returns):
        """Update forecast from period before."""
        raise NotImplementedError # pragma: no cover


@dataclass(unsafe_hash=True)
class HistoricalMeanReturn(BaseForecast):
    r"""Historical mean returns.

    This ignores both the cash returns column and all missing values.
    """

    def __post_init__(self):
        self._last_time = None
        self._last_counts = None
        self._last_sum = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.__post_init__()

    def values_in_time( # pylint: disable=arguments-differ
            self, t, past_returns, **kwargs):
        """Obtain current value of the mean returns.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past market returns, including cash.
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Means of past returns (excluding cash).
        :rtype: numpy.array
        """
        self._agnostic_update(t=t, past_returns=past_returns)
        return (self._last_sum / self._last_counts).values

    def _initial_compute(self, t, past_returns):
        """Make forecast from scratch."""
        self._last_counts = past_returns.iloc[:, :-1].count()
        self._last_sum = past_returns.iloc[:, :-1].sum()
        self._last_time = t

    def _online_update(self, t, past_returns):
        """Update forecast from period before."""
        self._last_counts += ~(past_returns.iloc[-1, :-1].isnull())
        self._last_sum += past_returns.iloc[-1, :-1].fillna(0.)
        self._last_time = t


@dataclass(unsafe_hash=True)
class HistoricalVariance(BaseForecast):
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
            self.meanforecaster = HistoricalMeanReturn()
        self._last_time = None
        self._last_counts = None
        self._last_sum = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.__post_init__()

    def values_in_time( # pylint: disable=arguments-differ
            self, t, past_returns, **kwargs):
        """Obtain current value either by update or from scratch.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past market returns, including cash.
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Variances of past returns (excluding cash).
        :rtype: numpy.array
        """
        self._agnostic_update(t=t, past_returns=past_returns)
        result = (self._last_sum / self._last_counts).values
        if not self.kelly:
            result -= self.meanforecaster.current_value**2
        return result

    def _initial_compute(self, t, past_returns):
        """Compute from scratch."""
        self._last_counts = past_returns.iloc[:, :-1].count()
        self._last_sum = (past_returns.iloc[:, :-1]**2).sum()
        self._last_time = t

    def _online_update(self, t, past_returns):
        """Update from estimate at t-1."""
        self._last_counts += ~(past_returns.iloc[-1, :-1].isnull())
        self._last_sum += past_returns.iloc[-1, :-1].fillna(0.)**2
        self._last_time = t


@dataclass(unsafe_hash=True)
class HistoricalStandardDeviation(HistoricalVariance):
    """Historical standard deviation."""

    kelly: bool = True

    def values_in_time(self, t, past_returns, **kwargs):
        """Obtain current value either by update or from scratch.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past market returns, including cash.
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviations of past returns (excluding cash).
        :rtype: numpy.array
        """
        variances = \
            super().values_in_time(t=t, past_returns=past_returns, **kwargs)
        return np.sqrt(variances)

class HistoricalMeanError(HistoricalVariance):
    r"""Historical standard deviations of the mean of non-cash returns.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2},
    \ldots, r_0` this is :math:`\sqrt{\text{Var}[r]/t}`. When there are
    missing values we ignore them, both to compute the variance and the
    count.
    """

    def __init__(self):
        super().__init__(kelly=False)

    def values_in_time(self, t, past_returns, **kwargs):
        """Obtain current value either by update or from scratch.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past market returns, including cash.
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviation of the mean of past returns (excluding
            cash).
        :rtype: numpy.array
        """
        variance = super().values_in_time(
            t=t, past_returns=past_returns, **kwargs)
        return np.sqrt(variance / self._last_counts.values)


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
class HistoricalFactorizedCovariance(BaseForecast):
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

    kelly: bool = True

    # this is used by FullCovariance
    FACTORIZED = True

    def __post_init__(self):
        self._last_time = None
        self._last_counts_matrix = None
        self._last_sum_matrix = None
        self._joint_mean = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.__post_init__()

    @staticmethod
    def _get_count_matrix(past_returns):
        r"""We obtain the matrix of non-null joint counts:

        .. math::

            \text{Count}\left(r^{i}r^{j} \neq \texttt{nan}\right).
        """
        tmp = (~past_returns.iloc[:, :-1].isnull()) * 1.
        return tmp.T @ tmp

    @staticmethod
    def _get_initial_joint_mean(past_returns):
        r"""Compute precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        nonnull = (~past_returns.iloc[:, :-1].isnull()) * 1.
        tmp = nonnull.T @ past_returns.iloc[:, :-1].fillna(0.)
        return tmp  # * tmp.T

    def _initial_compute(self, t, past_returns):
        self._last_counts_matrix = self._get_count_matrix(past_returns).values
        filled = past_returns.iloc[:, :-1].fillna(0.).values
        self._last_sum_matrix = filled.T @ filled
        if not self.kelly:
            self._joint_mean = self._get_initial_joint_mean(past_returns)

        self._last_time = t

    def _online_update(self, t, past_returns):
        nonnull = ~(past_returns.iloc[-1, :-1].isnull())
        self._last_counts_matrix += np.outer(nonnull, nonnull)
        last_ret = past_returns.iloc[-1, :-1].fillna(0.)
        self._last_sum_matrix += np.outer(last_ret, last_ret)
        self._last_time = t
        if not self.kelly:
            self._joint_mean += last_ret

    @online_cache
    def values_in_time( # pylint: disable=arguments-differ
            self, t, past_returns, **kwargs):
        """Obtain current value of the covariance estimate.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Extra arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.ForecastError: The procedure failed,
            typically because there are too many missing values (*e.g.*, some
            asset has only missing values).

        :returns: Square root factorized covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        self._agnostic_update(t=t, past_returns=past_returns)
        covariance = self._last_sum_matrix / self._last_counts_matrix

        if not self.kelly:
            tmp = self._joint_mean / self._last_counts_matrix
            covariance -= tmp.T * tmp
        try:
            return project_on_psd_cone_and_factorize(covariance)
        except np.linalg.LinAlgError as exc:
            raise ForecastError(f'Covariance estimation at time {t} failed;'
                + ' there are (probably) too many missing values in the'
                + ' past returns.') from exc
