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
"""This module contains classes that provide forecasts such as historical means.

and covariances and are used internally by cvxportfolio objects. In addition,
forecast classes have the ability to cache results online so that if multiple
classes need access to the estimated value (as is the case in MultiPeriodOptimization
policies) the expensive evaluation is only done once.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimator import PolicyEstimator


def online_cache(_values_in_time):
    """A simple online cache that decorates _values_in_time.

    The instance it is used on needs to be hashable (we currently
    use the hash of its __repr__ via dataclass).
    """

    def wrapped(self, t, cache=None, **kwargs):

        if cache is None:  # temporary to not change tests
            cache = {}

        if not (self in cache):
            cache[self] = {}

        if t in cache[self]:
            logging.debug(
                f'{self}._values_in_time at time {t} is retrieved from cache.')
            result = cache[self][t]
        else:
            result = _values_in_time(self, t=t, cache=cache, **kwargs)
            logging.debug(
                f'{self}._values_in_time at time {t} is stored in cache.')
            cache[self][t] = result
        return result

    return wrapped


class BaseForecast(PolicyEstimator):
    """Base class for forecasters."""

    # def _recursive_pre_evaluation(self, universe, backtest_times):
    #     self.universe = universe
    #     self.backtest_times = backtest_times
    #

    def _agnostic_update(self, t, past_returns):
        """Choose whether to make forecast from scratch or update last one."""
        if (self.last_time is None) or (self.last_time != past_returns.index[-1]):
            logging.debug(
                f'{self}._values_in_time at time {t} is computed from scratch.')
            self._initial_compute(t=t, past_returns=past_returns)
        else:
            logging.debug(
                f'{self}._values_in_time at time {t} is updated from previous value.')
            self._online_update(t=t, past_returns=past_returns)

    def _initial_compute(self, t, past_returns):
        """Make forecast from scratch."""
        raise NotImplementedError

    def _online_update(self, t, past_returns):
        """Update forecast from period before."""
        raise NotImplementedError


@dataclass(unsafe_hash=True)
class HistoricalMeanReturn(BaseForecast):
    r"""Historical mean returns.

    This ignores both the cash returns column and all missing values.
    """

    def __post_init__(self):
        self.last_time = None
        self.last_counts = None
        self.last_sum = None

    def _pre_evaluation(self, universe, backtest_times):
        self.__post_init__()

    def _values_in_time(self, t, past_returns, cache=None, **kwargs):
        self._agnostic_update(t=t, past_returns=past_returns)
        return (self.last_sum / self.last_counts).values

    def _initial_compute(self, t, past_returns):
        """Make forecast from scratch."""
        self.last_counts = past_returns.iloc[:, :-1].count()
        self.last_sum = past_returns.iloc[:, :-1].sum()
        self.last_time = t

    def _online_update(self, t, past_returns):
        """Update forecast from period before."""
        self.last_counts += ~(past_returns.iloc[-1, :-1].isnull())
        self.last_sum += past_returns.iloc[-1, :-1].fillna(0.)
        self.last_time = t


class HistoricalMeanError(BaseForecast):
    r"""Historical standard deviations of the mean of non-cash returns.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2}, \ldots, r_0` this is
    :math:`\sqrt{\text{Var}[r]/t}`. When there are missing values we ignore them,
    both to compute the variance and the count.
    """

    def __init__(self):
        self.varianceforecaster = HistoricalVariance(kelly=False)

    def _values_in_time(self, t, past_returns, **kwargs):
        return np.sqrt(self.varianceforecaster.current_value / self.varianceforecaster.last_counts.values)


@dataclass(unsafe_hash=True)
class HistoricalVariance(BaseForecast):
    r"""Historical variances of non-cash returns.

    :param kelly: if ``True`` compute :math:`\mathbf{E}[r^2]`, else
        :math:`\mathbf{E}[r^2] - {\mathbf{E}[r]}^2`. The second corresponds
        to the classic definition of variance, while the first is what is obtained
        by Taylor approximation of the Kelly gambling objective. (See page 28 of the book.)
    :type kelly: bool
    """

    kelly: bool = True

    def __post_init__(self):
        if not self.kelly:
            self.meanforecaster = HistoricalMeanReturn()
        self.last_time = None
        self.last_counts = None
        self.last_sum = None

    def _pre_evaluation(self, universe, backtest_times):
        self.__post_init__()

    def _values_in_time(self, t, past_returns, **kwargs):
        self._agnostic_update(t=t, past_returns=past_returns)
        result = (self.last_sum / self.last_counts).values
        if not self.kelly:
            result -= self.meanforecaster.current_value**2
        return result

    def _initial_compute(self, t, past_returns):
        self.last_counts = past_returns.iloc[:, :-1].count()
        self.last_sum = (past_returns.iloc[:, :-1]**2).sum()
        self.last_time = t

    # , last_estimation, last_counts, last_time):
    def _online_update(self, t, past_returns):
        self.last_counts += ~(past_returns.iloc[-1, :-1].isnull())
        self.last_sum += past_returns.iloc[-1, :-1].fillna(0.)**2
        self.last_time = t


@dataclass(unsafe_hash=True)
class HistoricalLowRankCovarianceSVD(BaseForecast):

    num_factors: int
    svd_iters: int = 10
    svd: str = 'numpy'

    # brought back from old commit
    # https://github.com/cvxgrp/cvxportfolio/commit/aa3d2150d12d85a6fb1befdf22cb7967fcc27f30
    # matches original 2016 method from example notebooks, with new heuristic for NaNs
    @staticmethod
    def build_low_rank_model(rets, num_factors=10, iters=10, svd='numpy'):
        r"""Build a low rank risk model from past returns that include NaNs.

        This is an experimental procedure that may work well on past returns
        matrices with few NaN values (say, below 20% of the total entries). 
        If there are (many) NaNs, one should probably also use a rather 
        large risk forecast error.
        """
        # rets = past_returns.iloc[:,:-1] # drop cash
        nan_fraction = rets.isnull().sum().sum() / np.prod(rets.shape)
        normalizer = np.sqrt((rets**2).mean())
        # if normalize:
        #     normalized = rets/(normalizer + 1E-8)
        # else:
        normalized = rets
        if nan_fraction:
            #if nan_fraction > 0.1 and not shrink:
            #    warnings.warn("Low rank model estimation on past returns with many NaNs should use the `shrink` option")
            nan_implicit_imputation = pd.DataFrame(0., columns=normalized.columns, index = normalized.index)
            for i in range(iters):
                if svd == 'numpy':
                    u, s, v = np.linalg.svd(normalized.fillna(nan_implicit_imputation), full_matrices=False)
                else:
                    raise Exception('Currently only numpy svd is implemented')
                nan_implicit_imputation = pd.DataFrame(
                    (u[:, :num_factors] * (s[:num_factors] #- s[num_factors] * shrink
                        )) @ v[:num_factors],
                    columns = normalized.columns, index = normalized.index)
        else:
            u, s, v = np.linalg.svd(normalized, full_matrices=False)
        F = v[:num_factors].T * s[:num_factors] / np.sqrt(len(rets))
        #if normalize:
        #    F = pd.DataFrame(F.T * (normalizer.values + 1E-8), columns=normalizer.index)
        #else:
        F = pd.DataFrame(F.T, columns=normalizer.index)
        idyosyncratic = normalizer**2 - (F**2).sum(0)
        if not np.all(idyosyncratic >= 0.):
            raise Exception("Low rank risk estimation with iterative SVD did not work.")
        return F.values, idyosyncratic.values

    @online_cache
    def _values_in_time(self, past_returns, **kwargs):

        return self.build_low_rank_model(past_returns.iloc[:, :-1],
            num_factors=self.num_factors,
            iters=self.svd_iters, svd=self.svd)


def project_on_psd_cone_and_factorize(Sigma):
    """Factorize matrix and remove negative eigenvalues."""
    eigval, eigvec = np.linalg.eigh(Sigma)
    eigval = np.maximum(eigval, 0.)
    return eigvec @ np.diag(np.sqrt(eigval))

@dataclass(unsafe_hash=True)
class HistoricalFactorizedCovariance(BaseForecast):
    r"""Historical covariance matrix, sqrt factorized.

    :param kelly: if ``True`` compute each :math:` \Sigma_{i,j} \simeq \mathbf{E}[r^{i} r^{j}]`, else
        :math:` \Sigma_{i,j} \simeq \mathbf{E}[r^{i} r^{j}] - \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]` matching
        the behavior of ``pandas.DataFrame.cov(ddof=0)`` (with same logic to handle missing data). 
        The second case corresponds to the classic definition of covariance, while the first is what is obtained
        by Taylor approximation of the Kelly gambling objective. (See page 28 of the book.)
    :type kelly: bool
    """

    kelly: bool = True

    # this is used by FullCovariance
    FACTORIZED = True

    def __post_init__(self):
        self.last_time = None

    def _pre_evaluation(self, universe, backtest_times):
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
        r"""Compute precursor of :math:`\Sigma_{i,j} = \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`.
        """
        nonnull = (~past_returns.iloc[:, :-1].isnull()) * 1.
        tmp = nonnull.T @ past_returns.iloc[:, :-1].fillna(0.)
        return tmp  # * tmp.T

    # @staticmethod
    # def _factorize(Sigma):
    #     """Factorize matrix and remove negative eigenvalues."""
    #     eigval, eigvec = np.linalg.eigh(Sigma)
    #     eigval = np.maximum(eigval, 0.)
    #     return eigvec @ np.diag(np.sqrt(eigval))

    def _initial_compute(self, t, past_returns):
        self.last_counts_matrix = self._get_count_matrix(past_returns).values
        filled = past_returns.iloc[:, :-1].fillna(0.).values
        self.last_sum_matrix = filled.T @ filled
        if not self.kelly:
            self.joint_mean = self._get_initial_joint_mean(past_returns)

        self.last_time = t

    def _online_update(self, t, past_returns):
        nonnull = ~(past_returns.iloc[-1, :-1].isnull())
        self.last_counts_matrix += np.outer(nonnull, nonnull)
        last_ret = past_returns.iloc[-1, :-1].fillna(0.)
        self.last_sum_matrix += np.outer(last_ret, last_ret)
        self.last_time = t
        if not self.kelly:
            self.joint_mean += last_ret

    @online_cache
    def _values_in_time(self, t, past_returns, **kwargs):

        self._agnostic_update(t=t, past_returns=past_returns)
        Sigma = self.last_sum_matrix / self.last_counts_matrix

        if not self.kelly:
            tmp = self.joint_mean / self.last_counts_matrix
            Sigma -= tmp.T * tmp

        return project_on_psd_cone_and_factorize(Sigma)
