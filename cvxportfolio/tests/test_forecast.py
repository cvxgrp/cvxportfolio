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
"""Unit tests for the data and parameter estimator objects."""

import unittest

import numpy as np
import pandas as pd

from cvxportfolio.forecast import (ForecastError,
                                   HistoricalFactorizedCovariance,
                                   HistoricalLowRankCovarianceSVD,
                                   HistoricalMeanError, HistoricalMeanReturn,
                                   HistoricalStandardDeviation,
                                   HistoricalVariance)
from cvxportfolio.tests import CvxportfolioTest


class TestForecast(CvxportfolioTest):
    """Test forecast estimators and their caching.
    
    In most cases we test against the relevant pandas function as reference.
    """

    def test_vector_fc_syntax(self):
        """Test syntax of vector forecasters."""
        with self.assertRaises(ValueError):
            forecaster = HistoricalMeanReturn(ma_window=3)
            forecaster.values_in_time_recursive(
                t=pd.Timestamp.today(), past_returns=self.returns)

        with self.assertRaises(ValueError):
            forecaster = HistoricalMeanReturn(ma_window=pd.Timedelta('0d'))
            forecaster.values_in_time_recursive(
                t=pd.Timestamp.today(), past_returns=self.returns)

        with self.assertRaises(ForecastError):
            forecaster = HistoricalMeanReturn()
            returns = pd.DataFrame(self.returns, copy=True)
            returns.iloc[:40, 3:10] = np.nan
            forecaster.values_in_time_recursive(
                t=self.returns.index[20], past_returns=returns.iloc[:20])

        # test that update throws exception when moving window results
        # in invalid data
        forecaster = HistoricalMeanReturn(ma_window=pd.Timedelta('10d'))
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[20:, 3:10] = np.nan
        last_valid_t = 25
        forecaster.values_in_time_recursive(
            t=self.returns.index[last_valid_t], 
            past_returns=returns.iloc[:last_valid_t])

        with self.assertRaises(ForecastError):
            forecaster.values_in_time_recursive(
                t=self.returns.index[last_valid_t+1], 
                past_returns=returns.iloc[:last_valid_t+1])

    def _base_test_vector_update(
            # pylint: disable=too-many-arguments
            self, forecaster, fc_kwargs, df_attr, pd_kwargs, df_callable=None):
        """Base function to test vector updates (mean, std, var)."""
        forecaster = forecaster(**fc_kwargs)
        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            cvx_val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)
            self.assertFalse(np.any(np.isnan(cvx_val)))
            self.assertTrue(np.allclose(
                cvx_val,
                df_callable(past_returns.iloc[:,:-1]) 
                    if df_callable is not None else
                    getattr(past_returns.iloc[:, :-1], df_attr)(**pd_kwargs)))

    def test_mean_update(self):
        """Test the mean forecaster."""
        self._base_test_vector_update(HistoricalMeanReturn, {}, 'mean', {})


    def test_variance_update(self):
        """Test the variance forecaster."""
        self._base_test_vector_update(
            HistoricalVariance, {'kelly':False}, 'var', {'ddof':0})

    def test_stddev_update(self):
        """Test the standard deviation forecaster."""
        self._base_test_vector_update(
            HistoricalStandardDeviation, {'kelly':False}, 'std', {'ddof':0})

    def test_meanerror_update(self):
        """Test the forecaster of the standard deviation on the mean."""

        def _me(past_returns_noncash):
            return past_returns_noncash.std(ddof=0)\
                / np.sqrt(past_returns_noncash.count())

        self._base_test_vector_update(
            HistoricalMeanError, {}, None, None, df_callable=_me)

    def _base_test_moving_window_vector_update(
            # pylint: disable=too-many-arguments
            self, forecaster, fc_kwargs, df_attr, pd_kwargs, df_callable=None):
        """Base test for vector quantities using a moving window."""

        window = pd.Timedelta('20d')
        forecaster = forecaster(**fc_kwargs, ma_window=window)
        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            past_returns_window = past_returns.loc[
                past_returns.index >= t - window]
            cvx_val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)

            self.assertFalse(np.any(np.isnan(cvx_val)))
            self.assertTrue(np.allclose(
                cvx_val,
                df_callable(past_returns_window.iloc[:,:-1]) 
                    if df_callable is not None else
                    getattr(past_returns_window.iloc[:, :-1], 
                        df_attr)(**pd_kwargs)))

    def test_mean_update_moving_window(self):
        """Test the mean forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalMeanReturn, {}, 'mean', {})

    def test_variance_update_moving_window(self):
        """Test the variance forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalVariance, {'kelly':False}, 'var', {'ddof':0})

    def test_stddev_update_moving_window(self):
        """Test the standard deviation forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalStandardDeviation, {'kelly':False}, 'std', {'ddof':0})

    def test_meanerror_update_moving_window(self):
        """Test standard deviation on the mean with moving window."""

        def _me(past_returns_noncash):
            return past_returns_noncash.std(ddof=0)\
                / np.sqrt(past_returns_noncash.count())

        self._base_test_moving_window_vector_update(
            HistoricalMeanError, {}, None, None, df_callable=_me)

    def _base_test_exponential_moving_window_vector_update(
            self, forecaster, fc_kwargs, df_callable=None):
        """Base test for vector quantities using an exponential moving window,
        and an exponential moving window in combination with a moving window.
        """
        half_life = pd.Timedelta('20d')
        inst_forecaster = forecaster(**fc_kwargs, ema_half_life=half_life)        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57, 58, 59, 60]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]

            cvx_val = inst_forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)

            self.assertFalse(np.any(np.isnan(cvx_val)))
            self.assertTrue(np.allclose(
                cvx_val, df_callable(past_returns.iloc[:,:-1], half_life)))

        # Test with both MA and EMA

        half_life = pd.Timedelta('10d')
        window = pd.Timedelta('20d')
        inst_forecaster = forecaster(
            **fc_kwargs, ma_window = window, ema_half_life=half_life)        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57, 58, 59, 60]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            past_returns_window = past_returns.loc[
                past_returns.index >= t - window]
            cvx_val = inst_forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)

            self.assertFalse(np.any(np.isnan(cvx_val)))
            self.assertTrue(np.allclose(
                cvx_val, df_callable(
                    past_returns_window.iloc[:,:-1], half_life)))

    @staticmethod
    def _mean_ewm(past_returns_noncash, half_life):
        return past_returns_noncash.ewm(
            halflife=half_life, times=past_returns_noncash.index
            ).mean().iloc[-1]

    def test_mean_update_exponential_moving_window(self):
        """Test the mean forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalMeanReturn, {}, self._mean_ewm)

    @staticmethod
    def _var_ewm(past_returns_noncash, half_life):
        """We need to do this b/c pandas.DataFrame.ewm.var doesn't support
        the ddof=0 option."""
        return (past_returns_noncash**2).ewm(
                halflife=half_life, times=past_returns_noncash.index
                ).mean().iloc[-1] - TestForecast._mean_ewm(
                    past_returns_noncash, half_life)**2

    def test_variance_update_exponential_moving_window(self):
        """Test the var forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalVariance, {'kelly':False}, self._var_ewm)

    def test_stddev_update_exponential_moving_window(self):
        """Test the std forecaster with exponential moving window."""

        def _std_ewm(*args):
            return np.sqrt(self._var_ewm(*args))
        self._base_test_exponential_moving_window_vector_update(
            HistoricalStandardDeviation, {'kelly':False}, _std_ewm)

    def test_counts_matrix(self):
        """Test internal method(s) of HistoricalFactorizedCovariance."""
        forecaster = HistoricalFactorizedCovariance()  # kelly=True)
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        returns.iloc[10:15, 10:20] = np.nan

        # pylint: disable=protected-access
        count_matrix = forecaster._compute_denominator(returns.iloc[:,:-1], None)

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18),
                (7, 24), (1, 15), (13, 22)]:
            print(count_matrix.iloc[indexes[0], indexes[1]])
            print(len((returns.iloc[:, indexes[0]] *
                  returns.iloc[:, indexes[1]]).dropna()))
            self.assertTrue(
                np.isclose(count_matrix.iloc[indexes[0], indexes[1]],
                    len((returns.iloc[:, indexes[0]]
                         * returns.iloc[:, indexes[1]]).dropna())))

    def test_sum_matrix(self):
        """Test internal method(s) of HistoricalFactorizedCovariance."""
        forecaster = HistoricalFactorizedCovariance()  # kelly=True)
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        returns.iloc[10:15, 10:20] = np.nan

        forecaster.values_in_time_recursive(
            t=pd.Timestamp('2022-01-01'), past_returns=returns)

        # pylint: disable=protected-access
        sum_matrix = forecaster._numerator

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18),
                (7, 24), (1, 15), (13, 22)]:
            print()
            print(sum_matrix.iloc[indexes[0], indexes[1]])
            print((returns.iloc[:, indexes[0]] *
                  returns.iloc[:, indexes[1]]).sum())
            self.assertTrue(np.isclose(
                sum_matrix.iloc[indexes[0], indexes[1]],
                (returns.iloc[:, indexes[0]] *
                 returns.iloc[:, indexes[1]]).sum()
            ))

    def test_covariance_update(self):
        """Test covariance forecast estimator."""

        forecaster = HistoricalFactorizedCovariance()

        returns = pd.DataFrame(self.returns.iloc[:, :4], copy=True)
        returns.iloc[:20, 1] = np.nan
        returns.iloc[10:30, 0] = np.nan
        returns.iloc[25:40, 2] = np.nan

        def _compute_covariance(rets):
            res = np.zeros((3, 3))
            res[0, 0] = np.nanmean(rets.iloc[:, 0] * rets.iloc[:, 0])
            res[1, 1] = np.nanmean(rets.iloc[:, 1] * rets.iloc[:, 1])
            res[2, 2] = np.nanmean(rets.iloc[:, 2] * rets.iloc[:, 2])
            res[0, 1] = np.nanmean(rets.iloc[:, 0] * rets.iloc[:, 1])
            res[0, 2] = np.nanmean(rets.iloc[:, 0] * rets.iloc[:, 2])
            res[1, 2] = np.nanmean(rets.iloc[:, 1] * rets.iloc[:, 2])
            res[1, 0] = res[0, 1]
            res[2, 0] = res[0, 2]
            res[2, 1] = res[1, 2]
            return res

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)
            Sigma = val @ val.T
            self.assertTrue(
                np.allclose(Sigma, _compute_covariance(past_returns)))

    def test_covariance_update_nokelly(self):
        """Test covariance forecast estimator.

        NOTE: due to a bug in pandas we can't test against
        pandas.DataFrame.cov,
        see https://github.com/pandas-dev/pandas/issues/45814 .
        In fact with the current bug in pandas
        ``past_returns.iloc[:,:-1].cov(ddof=0)`` returns
        ``past_returns.iloc[:,:-1].cov(ddof=1)``
        whenever there are missing values.
        """

        forecaster = HistoricalFactorizedCovariance(kelly=False)

        returns = pd.DataFrame(self.returns.iloc[:, :4], copy=True)
        returns.iloc[:20, 1] = np.nan
        returns.iloc[10:30, 0] = np.nan
        returns.iloc[25:40, 2] = np.nan

        def _cov_ij(i, j, rets):
            i_nanmasker = np.zeros(len(rets))
            i_nanmasker[rets.iloc[:, i].isnull()] = np.nan
            i_nanmasker[~(rets.iloc[:, i].isnull())] = 1.
            j_nanmasker = np.zeros(len(rets))
            j_nanmasker[rets.iloc[:, j].isnull()] = np.nan
            j_nanmasker[~(rets.iloc[:, j].isnull())] = 1.
            print(i_nanmasker, j_nanmasker)
            return np.nanmean(rets.iloc[:, i] * rets.iloc[:, j]
                ) - np.nanmean(rets.iloc[:, i] * j_nanmasker
                    ) * np.nanmean(rets.iloc[:, j] * i_nanmasker)

        def _compute_covariance(rets):
            res = np.zeros((3, 3))
            res[0, 0] = _cov_ij(0, 0, rets)
            res[1, 1] = _cov_ij(1, 1, rets)
            res[2, 2] = _cov_ij(2, 2, rets)
            res[0, 1] = _cov_ij(0, 1, rets)
            res[0, 2] = _cov_ij(0, 2, rets)
            res[1, 2] = _cov_ij(1, 2, rets)
            res[1, 0] = res[0, 1]
            res[2, 0] = res[0, 2]
            res[2, 1] = res[1, 2]
            return res

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)
            Sigma = val @ val.T

            self.assertTrue(
                np.allclose(Sigma, _compute_covariance(past_returns)))
            # pandasSigma = past_returns.iloc[:,:-1].cov(ddof=0)
            # self.assertTrue(np.allclose(Sigma, pandasSigma))
            self.assertTrue(np.allclose(
                np.diag(Sigma), past_returns.iloc[:, :-1].var(ddof=0)))

    def test_svd_forecaster(self):
        """Test the SVD forecaster.

        In particular, we compare it with the other covariance
        forecaster.

        We check that their forecasts are the same if there are no
        missing values in the past returns, and they diverge more as
        more missing values are introduced. Additionally, we check that
        it breaks when there are too many (more than 90%!) missing
        values.
        """

        returns = pd.DataFrame(self.returns.iloc[:, :4], copy=True)
        t = returns.index[10]

        num_factors = 1

        forecaster = HistoricalLowRankCovarianceSVD(
            num_factors=num_factors)
        forecaster2 = HistoricalFactorizedCovariance(kelly=True)

        def _compare_with_eigh(returns):

            F, d = forecaster.values_in_time_recursive(
                t=t, past_returns=returns)

            sigma_fact = forecaster2.values_in_time_recursive(
                t=t, past_returns=returns)

            sigma_svd = F.T @ F + np.diag(d)
            sigma_eigh = (
                sigma_fact[:, -num_factors:] @ sigma_fact[:, -num_factors:].T
                + np.diag((sigma_fact[:, :-num_factors]**2).sum(1)))
            print(sigma_svd)
            print(sigma_eigh)
            print(np.linalg.norm(sigma_svd),
                  np.linalg.norm(sigma_eigh),
                  np.linalg.norm(sigma_eigh-sigma_svd)
                  )

            forecaster3 = HistoricalLowRankCovarianceSVD(num_factors=1,
                svd='scipy')

            with self.assertRaises(SyntaxError):
                forecaster3.values_in_time_recursive(t=t, past_returns=returns)

            return np.linalg.norm(sigma_eigh-sigma_svd)

        self.assertTrue(np.isclose(_compare_with_eigh(returns), 0.))

        returns.iloc[:20, 1] = np.nan

        diff1 = _compare_with_eigh(returns)
        self.assertTrue(0 < diff1)

        returns.iloc[10:30, 0] = np.nan

        diff2 = _compare_with_eigh(returns)
        self.assertTrue(diff1 < diff2)

        returns.iloc[25:40, 2] = np.nan
        diff3 = _compare_with_eigh(returns)
        self.assertTrue(diff2 < diff3)

        print(returns.isnull().mean())
        returns.iloc[4:-3, -2] = np.nan
        print(returns.isnull().mean())
        diff4 = _compare_with_eigh(returns)
        self.assertTrue(diff3 < diff4)

        returns.iloc[:50, 0] = np.nan
        returns.iloc[50:, 1] = np.nan
        print(returns.isnull().mean())

        with self.assertRaises(ForecastError):
            forecaster2.values_in_time_recursive(t=t, past_returns=returns)

        # this one is even more robust!
        forecaster.values_in_time_recursive(
            t=t, past_returns=returns)

        returns.iloc[56:, 0] = np.nan
        print(returns.isnull().mean())

        forecaster.values_in_time_recursive(
            t=t, past_returns=returns)

        returns.iloc[:70, 1] = np.nan
        print(returns.isnull().mean())

        with self.assertRaises(ForecastError):
            forecaster.values_in_time_recursive(
                t=t, past_returns=returns)


if __name__ == '__main__':

    unittest.main() # pragma: no cover
