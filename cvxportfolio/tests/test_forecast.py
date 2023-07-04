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
from pathlib import Path


import cvxpy as cp
import numpy as np
import pandas as pd


from cvxportfolio.forecast import HistoricalMeanReturn, HistoricalMeanError, HistoricalVariance, HistoricalFactorizedCovariance


class TestEstimators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        # cls.sigma = pd.read_csv(Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(
            Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
        # cls.volumes = pd.read_csv(Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
        cls.w_plus = cp.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
        cls.z = cp.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]

    # def boilerplate(self, model):
    #     model._recursive_pre_evaluation(universe=self.returns.columns, backtest_times=self.returns.index)
    #     return model._compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)

    def test_mean_update(self):
        forecaster = HistoricalMeanReturn()  # lastforcash=True)

        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            mean = forecaster._recursive_values_in_time(
                t=t, past_returns=past_returns)
            # print(mean)
            # self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(
                mean, past_returns.iloc[:, :-1].mean()))

    def test_variance_update(self):
        forecaster = HistoricalVariance(kelly=False)

        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            var = forecaster._recursive_values_in_time(
                t=t, past_returns=past_returns)
            print(var)
            # self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(var, past_returns.var(ddof=0)[:-1]))

    def test_meanerror_update(self):
        forecaster = HistoricalMeanError()

        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            val = forecaster._recursive_values_in_time(
                t=t, past_returns=past_returns)
            print(val)
            # self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(val, past_returns.std(ddof=0)[
                            :-1] / np.sqrt(past_returns.count()[:-1])))

    def test_counts_matrix(self):
        forecaster = HistoricalFactorizedCovariance()  # kelly=True)
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        returns.iloc[10:15, 10:20] = np.nan

        count_matrix = forecaster._get_count_matrix(returns)

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18), (7, 24), (1, 15), (13, 22)]:
            print(count_matrix.iloc[indexes[0], indexes[1]])
            print(len((returns.iloc[:, indexes[0]] *
                  returns.iloc[:, indexes[1]]).dropna()))
            self.assertTrue(np.isclose(count_matrix.iloc[indexes[0], indexes[1]],
                                       len((returns.iloc[:, indexes[0]] * returns.iloc[:, indexes[1]]).dropna())))

    def test_sum_matrix(self):
        forecaster = HistoricalFactorizedCovariance()  # kelly=True)
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        returns.iloc[10:15, 10:20] = np.nan

        forecaster._recursive_values_in_time(
            t=pd.Timestamp('2022-01-01'), past_returns=returns)

        sum_matrix = forecaster.last_sum_matrix

        for indexes in [(1, 2), (4, 5), (1, 5), (7, 18), (7, 24), (1, 15), (13, 22)]:
            print()
            print(sum_matrix[indexes[0], indexes[1]])
            print((returns.iloc[:, indexes[0]] *
                  returns.iloc[:, indexes[1]]).sum())
            self.assertTrue(np.isclose(
                sum_matrix[indexes[0], indexes[1]],
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

        def compute_Sigma(rets):
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
            val = forecaster._recursive_values_in_time(
                t=t, past_returns=past_returns)
            Sigma = val @ val.T
            self.assertTrue(np.allclose(Sigma, compute_Sigma(past_returns)))

    def test_covariance_update_nokelly(self):
        """Test covariance forecast estimator.

        NOTE: due to a bug in pandas we can't test against pandas.DataFrame.cov, 
        see https://github.com/pandas-dev/pandas/issues/45814 . In fact with the
        current bug in pandas 
        ``past_returns.iloc[:,:-1].cov(ddof=0)`` returns ``past_returns.iloc[:,:-1].cov(ddof=1)``
        whenever there are missing values.
        """

        forecaster = HistoricalFactorizedCovariance(kelly=False)

        returns = pd.DataFrame(self.returns.iloc[:, :4], copy=True)
        returns.iloc[:20, 1] = np.nan
        returns.iloc[10:30, 0] = np.nan
        returns.iloc[25:40, 2] = np.nan

        def cov_ij(i, j, rets):
            i_nanmasker = np.zeros(len(rets))
            i_nanmasker[rets.iloc[:, i].isnull()] = np.nan
            i_nanmasker[~(rets.iloc[:, i].isnull())] = 1.
            j_nanmasker = np.zeros(len(rets))
            j_nanmasker[rets.iloc[:, j].isnull()] = np.nan
            j_nanmasker[~(rets.iloc[:, j].isnull())] = 1.
            print(i_nanmasker, j_nanmasker)
            return np.nanmean(rets.iloc[:, i] * rets.iloc[:, j]) - np.nanmean(rets.iloc[:, i] * j_nanmasker) * np.nanmean(rets.iloc[:, j] * i_nanmasker)

        def compute_Sigma(rets):
            res = np.zeros((3, 3))
            res[0, 0] = cov_ij(0, 0, rets)
            res[1, 1] = cov_ij(1, 1, rets)
            res[2, 2] = cov_ij(2, 2, rets)
            res[0, 1] = cov_ij(0, 1, rets)
            res[0, 2] = cov_ij(0, 2, rets)
            res[1, 2] = cov_ij(1, 2, rets)
            res[1, 0] = res[0, 1]
            res[2, 0] = res[0, 2]
            res[2, 1] = res[1, 2]
            return res

        for tidx in [50, 51, 52, 55, 56, 57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index < t]
            val = forecaster._recursive_values_in_time(
                t=t, past_returns=past_returns)
            Sigma = val @ val.T

            self.assertTrue(np.allclose(Sigma, compute_Sigma(past_returns)))
            # pandasSigma = past_returns.iloc[:,:-1].cov(ddof=0)
            # self.assertTrue(np.allclose(Sigma, pandasSigma))
            self.assertTrue(np.allclose(
                np.diag(Sigma), past_returns.iloc[:, :-1].var(ddof=0)))


if __name__ == '__main__':
    unittest.main()
