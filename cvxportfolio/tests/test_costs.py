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

"""Unit tests for the cost objects."""

import unittest
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

import cvxportfolio as cvx

from cvxportfolio.costs import *
from cvxportfolio.returns import *
from cvxportfolio.risks import *
# from cvxportfolio.legacy import LegacyReturnsForecast #, MultipleReturnsForecasts


class TestCosts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        # cls.sigma = pd.read_csv(Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(
            Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
        cls.volumes = pd.read_csv(
            Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
        cls.w_plus = cp.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
        cls.z = cp.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]

    def test_cost_algebra(self):
        # n = len(self.returns.columns)
        # wplus = cvx.Variable(n)
        self.w_plus_minus_w_bm.value = np.random.uniform(size=self.N)
        self.w_plus_minus_w_bm.value /= sum(self.w_plus_minus_w_bm.value)
        t = self.returns.index[1]

        cost1 = -.5 * DiagonalCovariance()
        cost2 = -.5 * FullCovariance(
            self.returns.iloc[:, :-1].T @ self.returns.iloc[:, :-1] / len(self.returns))
        cost3 = cost1 + cost2

        cost3._recursive_pre_evaluation(
            universe=self.returns.columns, backtest_times=self.returns.index)
        expr3 = cost3._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        expr1 = cost1._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        expr2 = cost2._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        cost3._recursive_values_in_time(
            t=t, past_returns=self.returns.loc[self.returns.index < t])
        self.assertTrue(expr3.value == expr1.value + expr2.value)

        cost4 = cost1 * 2
        expr4 = cost4._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        self.assertTrue(expr4.value == expr1.value * 2)

        cost3 = cost1 + 3 * cost2
        expr3 = cost3._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        self.assertTrue(expr3.value == expr1.value + 3 * expr2.value)

        cost3 = 3 * cost1 + 2 * cost2
        expr3 = cost3._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        self.assertTrue(expr3.value == 3 * expr1.value + 2 * expr2.value)

        cost3 = .1 * cost1 + 2 * (cost2 + cost1)
        expr3 = cost3._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        self.assertTrue(np.isclose(expr3.value, .1 * expr1.value +
                        2 * (expr2.value + expr1.value)))

        cost3 = cost1 + 5 * (cost2 + cost1)
        expr3 = cost3._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        self.assertTrue(np.isclose(expr3.value, expr1.value +
                        5 * (expr2.value + expr1.value)))

    def test_hcost(self):
        """Test holding cost model."""
        dividends = pd.Series(np.random.randn(self.N-1),
                              self.returns.columns[:-1])
        dividends *= 0.
        hcost = HoldingCost(short_fees=5, dividends=dividends)

        t = 100  # this is picked so that periods_per_year evaluates to 252
        hcost._recursive_pre_evaluation(
            universe=self.returns.columns, backtest_times=self.returns.index)
        expression = hcost._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)
        hcost._recursive_values_in_time(
            t=self.returns.index[t], past_returns=self.returns.iloc[:t])
        cash_ret = self.returns.iloc[t-1].iloc[-1]

        for i in range(10):
            self.w_plus.value = np.random.randn(self.N)
            self.w_plus.value[-1] = 1 - np.sum(self.w_plus.value[:-1])

            print(expression.value)

            # print(-np.sum(np.minimum(self.w_plus.value[:-1], 0.)) * (
            #     cash_ret + 5/(100 * 252)))
            # print(- self.w_plus.value[:-1].T @ dividends)
            # print(-np.sum(np.minimum(self.w_plus.value[:-1], 0.)) * (cash_ret + 0.5/(100 * 252))
            #       - self.w_plus.value[:-1].T @ dividends)

            self.assertTrue(np.isclose(expression.value,
                                       -np.sum(np.minimum(self.w_plus.value[:-1], 0.)) * (np.exp(np.log(1.05)/252) - 1)
                                       # + np.abs(self.w_plus.value[-1])* 0.5/(100 * 252)
                                       - self.w_plus.value[:-1].T @ dividends
                                       ))

    def test_tcost(self):
        """Test tcost model."""
        value = 1e6

        pershare_cost = pd.Series([0., 0.005, 0.], [
                                  self.returns.index[12], self.returns.index[23], self.returns.index[34]])
        b = pd.Series([0., 0., 1.], [self.returns.index[12],
                      self.returns.index[23], self.returns.index[34]])

        tcost = StocksTransactionCost(
            a=0.001/2, pershare_cost=pershare_cost, b=b, window_sigma_est=250, window_volume_est=250, exponent=1.5)

        t = self.returns.index[12]

        tcost._recursive_pre_evaluation(
            universe=self.returns.columns, backtest_times=self.returns.index)
        expression = tcost._compile_to_cvxpy(
            self.w_plus, self.z, self.w_plus_minus_w_bm)

        # only spread

        tcost._recursive_values_in_time(t=self.returns.index[12],
                                        current_portfolio_value=value,
                                        past_returns=self.returns.iloc[:12],
                                        past_volumes=self.volumes.iloc[:12],
                                        current_prices=pd.Series(np.ones(self.returns.shape[1]-1), self.returns.columns[:-1]))

        self.z.value = np.random.randn(self.returns.shape[1])
        self.z.value[-1] = -np.sum(self.z.value[:-1])

        est_tcost_lin = sum(np.abs(self.z.value[:-1]) * 0.0005)
        print(est_tcost_lin)
        print(expression.value)
        self.assertTrue(np.isclose(expression.value, est_tcost_lin))

        # spread and fixed cost

        prices = pd.Series(np.random.uniform(
            1, 100, size=self.returns.shape[1]-1), self.returns.columns[:-1])

        tcost._recursive_values_in_time(t=self.returns.index[23],
                                        current_portfolio_value=value,
                                        past_returns=self.returns.iloc[:23],
                                        past_volumes=self.volumes.iloc[:23],
                                        current_prices=prices)

        self.z.value = np.random.randn(self.returns.shape[1])
        self.z.value[-1] = -np.sum(self.z.value[:-1])

        est_tcost_lin = sum(np.abs(self.z.value[:-1]) * 0.0005)
        est_tcost_lin += np.abs(self.z.value[:-1]) @ (0.005 / prices)
        print(est_tcost_lin)
        print(expression.value)
        self.assertTrue(np.isclose(expression.value, est_tcost_lin))

        # spread and nonlin cost

        tcost._recursive_values_in_time(t=self.returns.index[34],
                                        current_portfolio_value=value,
                                        past_returns=self.returns.iloc[:34],
                                        past_volumes=self.volumes.iloc[:34],
                                        current_prices=pd.Series(np.ones(self.returns.shape[1]-1), self.returns.columns[:-1]))

        self.z.value = np.random.randn(self.returns.shape[1])
        self.z.value[-1] = -np.sum(self.z.value[:-1])

        est_tcost_lin = sum(np.abs(self.z.value[:-1]) * 0.0005)
        volumes_est = self.volumes.iloc[:34].mean().values
        sigmas_est = np.sqrt((self.returns.iloc[:34, :-1]**2).mean()).values
        est_tcost_nonnlin = (
            np.abs(self.z.value[:-1])**(3/2)) @ (sigmas_est * np.sqrt(value / volumes_est))
        print(est_tcost_lin)
        print(est_tcost_nonnlin)
        print(expression.value)
        self.assertTrue(np.isclose(expression.value,
                        est_tcost_lin+est_tcost_nonnlin))


if __name__ == '__main__':
    unittest.main()
