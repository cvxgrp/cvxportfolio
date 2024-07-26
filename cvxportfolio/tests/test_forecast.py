# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for the data and parameter estimator objects."""

import copy
import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.forecast import (ForecastError, HistoricalCovariance,
                                   HistoricalFactorizedCovariance,
                                   HistoricalLowRankCovarianceSVD,
                                   HistoricalMeanError, HistoricalMeanReturn,
                                   HistoricalMeanVolume,
                                   HistoricalStandardDeviation,
                                   HistoricalVariance, RegressionMeanReturn,
                                   RegressionXtYReturns, UserProvidedRegressor)
from cvxportfolio.tests import CvxportfolioTest
from cvxportfolio.utils import set_pd_read_only


class TestForecast(CvxportfolioTest): # pylint: disable=too-many-public-methods
    """Test forecast estimators and their caching.

    In most cases we test against the relevant pandas function as reference.
    """

    @classmethod
    def setUpClass(cls):
        """Add a few things used in tests here."""
        super().setUpClass()
        cls.aligned_regressor = pd.Series(
            np.random.randn(len(cls.market_data.returns)), cls.returns.index)
        cls.aligned_regressor.name = 'regressor_aligned'

        # this one has index 1h before each trading period, and starts later
        cls.unaligned_regressor = pd.Series(
            np.random.randn(len(cls.market_data.returns)-100),
            cls.returns.index[100:] - pd.Timedelta('3600s'))
        cls.unaligned_regressor.name = 'regressor_unaligned'

    def test_estimate(self):
        """Test estimate method of a forecaster."""
        forecaster = cvx.forecast.HistoricalCovariance()

        forecaster.estimate(
            market_data=self.market_data,
            t=self.market_data.trading_calendar()[20])

        forecaster = cvx.forecast.HistoricalCovariance(kelly=False)
        t_fore = self.market_data.trading_calendar()[-1]
        full_cov = forecaster.estimate(
            market_data=self.market_data,
            t=t_fore)

        pdcov = self.market_data.returns.loc[
            self.market_data.returns.index < t_fore].iloc[:, :-1].cov(ddof=0)

        self.assertTrue(np.allclose(full_cov, pdcov))

        with self.assertRaises(ValueError):
            forecaster.estimate(
                market_data=self.market_data,
                t = self.market_data.returns.index[-1] + pd.Timedelta('300d'))

    def test_nested_cached_eval(self):
        """Test nested evaluation with caching."""

        forecaster = HistoricalVariance(kelly=False)
        forecaster._CACHED = True # pylint: disable=protected-access
        md = self.market_data
        returns = md.returns
        cache = {}
        for tidx in [-30, -29, -25, -24, -23, -30, -29]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]

            result = forecaster.values_in_time_recursive(
                    past_returns=past_returns, t=t, cache=cache)

            self.assertTrue(
                np.allclose(result, past_returns.iloc[:, :-1].var(ddof=0)))
        self.assertEqual(len(cache), 1)
        self.assertEqual(len(list(cache.values())[0]), 5)

    @staticmethod
    def _simple_aligner(regressor, index):
        """Align regressor on DatetimeIndex."""
        return regressor.reindex(index, method='ffill').dropna()

    @staticmethod
    def _simple_kelly_cov(past_returns):
        """Same logic as cov, used to test XtX regression matrix."""
        filled = past_returns.fillna(0.)
        nonnull = (~past_returns.isnull()) * 1.
        num = filled.T @ filled
        den = nonnull.T @ nonnull
        return pd.DataFrame(
            num/den, index=past_returns.columns, columns=past_returns.columns)

    def _xtx_matrix(self, stock_returns, at_time, regressors):
        """What the code should give."""
        past_returns = stock_returns.loc[
            stock_returns.index < at_time].dropna()
        aligned_regressors = [
            self._simple_aligner(r, past_returns.index)
                for r in regressors]
        aligned_regressors = pd.DataFrame(aligned_regressors).T
        aligned_regressors['intercept'] = 1.
        return self._simple_kelly_cov(aligned_regressors)

    def test_regression_XtX_matrices(self):
        """Test one of the components of returns regression."""
        md = copy.deepcopy(self.market_data)
        regressors = [self.aligned_regressor, self.unaligned_regressor]

        self._xtx_matrix(md.returns['AAPL'], md.returns.index[-30], regressors)

        # TODO, finish this

    def test_regression_xty_returns(self):
        """Test one of the components of returns regression."""
        md = copy.deepcopy(self.market_data)
        for regressor in [self.aligned_regressor, self.unaligned_regressor]:
            xty = RegressionXtYReturns(
                regressor=UserProvidedRegressor(regressor))
            t_fore = md.returns.index[-3]

            # check that estimate is correct
            my_estimate = xty.estimate(md, t=t_fore)
            pd_estimate = md.returns.iloc[:, :-1].multiply(
                self._simple_aligner(regressor, md.returns.index),
                    axis=0).loc[md.returns.index < t_fore].mean()
            self.assertTrue(np.allclose(my_estimate, pd_estimate))

            # iteratively
            xty.initialize_estimator_recursive(
                universe=md.returns.columns, trading_calendar=md.returns.index)

            for tidx in [-30, -29, -25, -24, -23]:
                t = md.returns.index[tidx]
                past_returns = md.returns.loc[md.returns.index < t]

                my_result  = \
                    xty.values_in_time_recursive(
                        past_returns=past_returns, t=t)
                pd_result = md.returns.iloc[:, :-1].multiply(
                    self._simple_aligner(regressor, md.returns.index),
                        axis=0).loc[md.returns.index < t].mean()
                self.assertTrue(np.allclose(my_result, pd_result))

            xty.finalize_estimator_recursive()

    @unittest.expectedFailure # code for this being redesigned
    def test_regression_mean_return(self): # pragma: no cover
         # pylint: disable=too-many-locals
        """Test historical mean return with regression."""

        # will be refactored
        # vix = cvx.YahooFinance('^VIX').data.open
        # vix.name = 'VIX'

        md = copy.deepcopy(self.market_data)
        rets = pd.DataFrame(self.market_data.returns, copy=True)
        rets.loc[rets.index <= rets.index[6], 'AAPL'] = np.nan
        rets.loc[rets.index <= rets.index[5], 'WMT'] = np.nan
        assert rets.iloc[0].isnull().sum() > 0
        md.returns = set_pd_read_only(rets)

        np.random.seed(0)
        vix = pd.Series(np.random.uniform(len(md.returns)), md.returns.index)
        vix.name = 'VIX'
        regr_mean_ret = RegressionMeanReturn(regressors=[vix])
        regr_mean_ret.initialize_estimator_recursive(
            universe=self.market_data.returns.columns,
            trading_calendar=self.market_data.returns.index)
        regr_mean_ret._CACHED = True # pylint: disable=protected-access
        returns = md.returns

        # ValueError thrown by UserProvidedRegressor (not enough history)
        with self.assertRaises(ValueError):
            t = returns.index[9]
            past_returns = returns.loc[returns.index < t]
            regr_mean_ret.values_in_time_recursive(
                past_returns=past_returns, t=t)

        for tidx in [-30, -29, -25, -24, -23]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]

            result  = \
                regr_mean_ret.values_in_time_recursive(
                    past_returns=past_returns, t=t)
            # print('result')
            # print(result)

            # reproduce here
            for asset in ['AAPL', 'WMT', 'MMM']:
                y = past_returns[asset].dropna()
                x = pd.DataFrame(1., index=y.index, columns=['intercept'])
                x['VIX'] = vix.reindex(y.index, method='ffill')
                beta = np.linalg.solve(x.T @ x, x.T @ y)
                x_last = pd.Series(1., index=['intercept'])
                x_last['VIX'] = vix[vix.index < t].iloc[-1]
                local_result = x_last @ beta
                # print(local_result)
                # print(result[asset])
                self.assertTrue(np.isclose(local_result, result[asset]))

        # so it also calls finalize_estimator
        regr_mean_ret.estimate(market_data=md, t=md.returns.index[30])

    def test_historical_mean_volume(self):
        """Test mean volume forecaster."""

        forecaster = HistoricalMeanVolume()
        for tidx in [20, 21, 22, 25, 26, 27]:
            cvx_val = forecaster.values_in_time_recursive(
                t=self.volumes.index[tidx],
                past_returns=self.returns.iloc[:tidx],
                past_volumes=self.volumes.iloc[:tidx])

            self.assertTrue(np.allclose(cvx_val,
                self.volumes.iloc[:tidx].mean()))

        with self.assertRaises(ValueError):
            cvx_val = forecaster.values_in_time_recursive(
                t=self.volumes.index[tidx],
                past_returns=self.returns.iloc[:tidx],
                past_volumes=None)

    def test_vector_fc_syntax(self):
        """Test syntax of vector forecasters."""
        with self.assertRaises(ValueError):
            forecaster = HistoricalMeanReturn(rolling=3)
            forecaster.values_in_time_recursive(
                t=pd.Timestamp.today(), past_returns=self.returns)

        with self.assertRaises(ValueError):
            forecaster = HistoricalMeanReturn(rolling=pd.Timedelta('0d'))
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
        forecaster = HistoricalMeanReturn(rolling=pd.Timedelta('10d'))
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
            self, forecaster, fc_kwargs, df_attr, pd_kwargs, df_callable=None,
            with_nans=True):
        """Base function to test vector updates (mean, std, var)."""
        forecaster = forecaster(**fc_kwargs)

        returns = pd.DataFrame(self.returns, copy=True)
        if with_nans:
            returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            cvx_val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)
            self.assertFalse(np.any(np.isnan(cvx_val)))
            self.assertTrue(np.allclose(
                cvx_val,
                df_callable(past_returns.iloc[:, :-1])
                    if df_callable is not None else
                    getattr(past_returns.iloc[:, :-1], df_attr)(**pd_kwargs)))

    def test_mean_update(self):
        """Test the mean forecaster."""
        self._base_test_vector_update(HistoricalMeanReturn, {}, 'mean', {})

    def test_variance_update(self):
        """Test the variance forecaster."""
        self._base_test_vector_update(
            HistoricalVariance, {'kelly': False}, 'var', {'ddof': 0})

    def test_stddev_update(self):
        """Test the standard deviation forecaster."""
        self._base_test_vector_update(
            HistoricalStandardDeviation, {'kelly': False}, 'std', {'ddof': 0})

    def test_meanerror_update(self):
        """Test the forecaster of the standard deviation on the mean."""

        def _me(past_returns_noncash):
            return past_returns_noncash.std(ddof=0)\
                / np.sqrt(past_returns_noncash.count())

        self._base_test_vector_update(
            HistoricalMeanError, {}, None, None, df_callable=_me)

    def _kelly_covariance(self, past_returns_noncash, half_life=None):
        """Covariance using Kelly.

        Below we implement also the one not using Kelly. That should be
        done by past_returns_noncash.cov(ddof=0) but a bug in pandas
        prevents from doing so (see below).
        """

        # EMW
        if half_life is not None:
            index_in_halflifes = (
                past_returns_noncash.index - past_returns_noncash.index[-1]
                    ) / half_life
            emw_weights = np.exp(index_in_halflifes * np.log(2))
        else:
            emw_weights = pd.Series(1., past_returns_noncash.index)

        _ = past_returns_noncash.fillna(0.)
        num = _.T @ _.multiply(emw_weights, axis=0)
        _ = (~past_returns_noncash.isnull()) * 1.
        den = _.T @ _.multiply(emw_weights, axis=0)

        return num / den

    def _nokelly_covariance_emw_nonans(self, past_returns_noncash, half_life):
        """This is only without nans."""
        result = self._kelly_covariance(
            past_returns_noncash, half_life=half_life)
        means = self._mean_emw(past_returns_noncash, half_life)
        return result - np.outer(means, means)

    def test_cov_update(self):
        """Test the covariance forecaster."""

        self._base_test_vector_update(
            HistoricalCovariance, {'kelly': True}, None, None,
            df_callable=self._kelly_covariance)

    def test_cov_update_nokelly(self):
        """Test the covariance forecaster without Kelly correction.

        Due to a bug in pandas, we can compare with Pandas' DataFrame.cov
        only if the df has no NaNs.
        """

        self._base_test_vector_update(
            HistoricalCovariance, {'kelly': False}, 'cov', {'ddof': 0},
            with_nans=False)

    def _base_test_moving_window_vector_update(
            # pylint: disable=too-many-arguments
            self, forecaster, fc_kwargs, df_attr, pd_kwargs, df_callable=None,
            with_nans=True):
        """Base test for vector quantities using a moving window."""

        window = pd.Timedelta('20d')
        forecaster = forecaster(**fc_kwargs, rolling=window)

        returns = pd.DataFrame(self.returns, copy=True)
        if with_nans:
            returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            past_returns_window = past_returns.loc[
                past_returns.index >= t - window]
            cvx_val = forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)

            self.assertFalse(np.any(np.isnan(cvx_val)))

            test_val = (df_callable(past_returns_window.iloc[:, :-1])
                    if df_callable is not None else
                    getattr(past_returns_window.iloc[:, :-1],
                        df_attr)(**pd_kwargs))

            self.assertTrue(np.allclose(cvx_val, test_val))

    def test_mean_update_moving_window(self):
        """Test the mean forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalMeanReturn, {}, 'mean', {})

    def test_variance_update_moving_window(self):
        """Test the variance forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalVariance, {'kelly': False}, 'var', {'ddof': 0})

    def test_stddev_update_moving_window(self):
        """Test the standard deviation forecaster with moving window."""
        self._base_test_moving_window_vector_update(
            HistoricalStandardDeviation, {'kelly': False}, 'std', {'ddof': 0})

    def test_meanerror_update_moving_window(self):
        """Test standard deviation on the mean with moving window."""

        def _me(past_returns_noncash):
            return past_returns_noncash.std(ddof=0)\
                / np.sqrt(past_returns_noncash.count())

        self._base_test_moving_window_vector_update(
            HistoricalMeanError, {}, None, None, df_callable=_me)

    def test_cov_update_moving_window(self):
        """Test the covariance forecaster  with moving window."""

        self._base_test_moving_window_vector_update(
            HistoricalCovariance, {'kelly': True}, None, None,
            df_callable=self._kelly_covariance)

    def test_cov_update_moving_window_nokelly(self):
        """Test the covariance forecaster with moving window without Kelly."""

        self._base_test_moving_window_vector_update(
            HistoricalCovariance, {'kelly': False}, 'cov', {'ddof': 0},
            with_nans=False)

    def _base_test_exponential_moving_window_vector_update(
            self, forecaster, fc_kwargs, df_callable=None, with_nans=True):
        """Base test for vector quantities w/ exponential moving window,
        and exponential moving window + moving window."""
        half_life = pd.Timedelta('20d')
        inst_forecaster = forecaster(**fc_kwargs, half_life=half_life)
        returns = pd.DataFrame(self.returns, copy=True)
        if with_nans:
            returns.iloc[:40, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57, 58, 59, 60]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]

            cvx_val = inst_forecaster.values_in_time_recursive(
                t=t, past_returns=past_returns)

            self.assertFalse(np.any(np.isnan(cvx_val)))

            # print(cvx_val - df_callable(past_returns.iloc[:,:-1], half_life))
            self.assertTrue(np.allclose(
                cvx_val, df_callable(past_returns.iloc[:, :-1], half_life)))

        # Test with both MA and EMA

        half_life = pd.Timedelta('10d')
        window = pd.Timedelta('20d')
        inst_forecaster = forecaster(
            **fc_kwargs, rolling = window, half_life=half_life)
        returns = pd.DataFrame(self.returns, copy=True)
        if with_nans:
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
                    past_returns_window.iloc[:, :-1], half_life)))

    @staticmethod
    def _mean_emw(past_returns_noncash, half_life):
        return past_returns_noncash.ewm(
            halflife=half_life, times=past_returns_noncash.index
            ).mean().iloc[-1]

    def test_mean_update_exponential_moving_window(self):
        """Test the mean forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalMeanReturn, {}, self._mean_emw)

    @staticmethod
    def _var_emw(past_returns_noncash, half_life):
        """We need to do this b/c pandas.DataFrame.emw.var doesn't support
        the ddof=0 option."""
        return (past_returns_noncash**2).ewm(
                halflife=half_life, times=past_returns_noncash.index
                ).mean().iloc[-1] - TestForecast._mean_emw(
                    past_returns_noncash, half_life)**2

    def test_variance_update_exponential_moving_window(self):
        """Test the var forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalVariance, {'kelly': False}, self._var_emw)

    def test_stddev_update_exponential_moving_window(self):
        """Test the std forecaster with exponential moving window."""
        def _std_emw(*args):
            return np.sqrt(self._var_emw(*args))
        self._base_test_exponential_moving_window_vector_update(
            HistoricalStandardDeviation, {'kelly': False}, _std_emw)

    def test_cov_update_exponential_moving_window(self):
        """Test the covariance forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalCovariance, {'kelly': True},
            df_callable=self._kelly_covariance)

    def test_cov_update_exponential_moving_window_nokelly(self):
        """Test the covariance forecaster with exponential moving window."""

        self._base_test_exponential_moving_window_vector_update(
            HistoricalCovariance, {'kelly': False},
            df_callable=self._nokelly_covariance_emw_nonans, with_nans=False)

    def test_counts_matrix(self):
        """Test internal method(s) of HistoricalFactorizedCovariance."""
        forecaster = HistoricalFactorizedCovariance()  # kelly=True)
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        returns.iloc[10:15, 10:20] = np.nan

        forecaster.values_in_time_recursive(
            t=pd.Timestamp('2022-01-01'), past_returns=returns)

        # pylint: disable=protected-access
        count_matrix = forecaster._denominator.current_value

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18),
                (7, 24), (1, 15), (13, 22)]:
            # print(count_matrix.iloc[indexes[0], indexes[1]])
            # print(len((returns.iloc[:, indexes[0]] *
            #       returns.iloc[:, indexes[1]]).dropna()))
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
        sum_matrix = forecaster._numerator.current_value

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18),
                (7, 24), (1, 15), (13, 22)]:
            # print()
            # print(sum_matrix.iloc[indexes[0], indexes[1]])
            # print((returns.iloc[:, indexes[0]] *
            #       returns.iloc[:, indexes[1]]).sum())
            self.assertTrue(np.isclose(
                sum_matrix.iloc[indexes[0], indexes[1]],
                (returns.iloc[:, indexes[0]] *
                 returns.iloc[:, indexes[1]]).sum()
            ))

    def test_covariance_update_withnans(self):
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

    def test_covariance_update_nokelly_withnans(self):
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
            # print(i_nanmasker, j_nanmasker)
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
            # print(sigma_svd)
            # print(sigma_eigh)
            # print(np.linalg.norm(sigma_svd),
            #      np.linalg.norm(sigma_eigh),
            #      np.linalg.norm(sigma_eigh-sigma_svd)
            #      )

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

        # print(returns.isnull().mean())
        returns.iloc[4:-3, -2] = np.nan
        # print(returns.isnull().mean())
        diff4 = _compare_with_eigh(returns)
        self.assertTrue(diff3 < diff4)

        returns.iloc[:50, 0] = np.nan
        returns.iloc[50:, 1] = np.nan
        # print(returns.isnull().mean())

        with self.assertRaises(ForecastError):
            forecaster2.values_in_time_recursive(t=t, past_returns=returns)

        # this one is even more robust!
        forecaster.values_in_time_recursive(
            t=t, past_returns=returns)

        returns.iloc[56:, 0] = np.nan
        # print(returns.isnull().mean())

        forecaster.values_in_time_recursive(
            t=t, past_returns=returns)

        returns.iloc[:70, 1] = np.nan
        # print(returns.isnull().mean())

        with self.assertRaises(ForecastError):
            forecaster.values_in_time_recursive(
                t=t, past_returns=returns)


if __name__ == '__main__': # pragma: no cover
    import logging
    logging.basicConfig(level='DEBUG')
    unittest.main(warnings='error')
