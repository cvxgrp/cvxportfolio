# Copyright (C) 2017-2024 Enzo Busseti
# Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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
#
## Earlier versions of this module had the following copyright and licensing
## notice, which is subsumed by the above.
##
### Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###    http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
"""Unit tests for the BacktestResult class and methods."""

import logging
import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.tests import CvxportfolioTest


class TestResult(CvxportfolioTest):
    """Test BacktestResult class and methods."""

    def test_backtest_with_time_changing_universe(self):
        """Test back-test with user-defined time-varying universe."""
        rets = pd.DataFrame(self.returns.iloc[:, -10:], copy=True)
        volumes = pd.DataFrame(self.volumes.iloc[:, -9:], copy=True)
        prices = pd.DataFrame(self.prices.iloc[:, -9:], copy=True)

        universe_selection = pd.DataFrame(
            True, index=rets.index, columns=rets.columns[:-1])
        universe_selection.iloc[14:25, 1:3] = False
        universe_selection.iloc[9:17, 3:5] = False
        universe_selection.iloc[8:15, 5:7] = False
        universe_selection.iloc[16:29, 7:8] = False
        # print(universe_selection.iloc[10:20])

        modified_market_data = cvx.UserProvidedMarketData(
            returns=rets, volumes=volumes, prices=prices,
            cash_key='cash',
            min_history=pd.Timedelta('0d'),
            universe_selection_in_time=universe_selection)

        simulator = cvx.StockMarketSimulator(
            market_data=modified_market_data,
            base_location=self.datadir)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - 10 * cvx.FullCovariance(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_socp_solver)

        # with self.assertRaises(ValueError):
        #     simulator.backtest(policy, start_time = rets.index[10],
        #     end_time = rets.index[9])

        bt_result = simulator.backtest(policy, start_time = rets.index[10],
            end_time = rets.index[20])

        # print(bt_result.w)

        self.assertTrue(set(bt_result.w.columns) == set(rets.columns))
        self.assertTrue(
            np.all(bt_result.w.iloc[:-1, :-1].isnull()
                == ~universe_selection.iloc[10:20]))

        # try without repeating the uni
        reduced = pd.DataFrame(universe_selection.iloc[10:20], copy=True)
        reduced = pd.DataFrame(reduced.iloc[[0, 4, 5, 6, 7]], copy=True)
        # print(reduced)

        modified_market_data = cvx.UserProvidedMarketData(
            returns=rets, volumes=volumes, prices=prices,
            cash_key='cash',
            min_history=pd.Timedelta('0d'),
            universe_selection_in_time=reduced)

        simulator = cvx.StockMarketSimulator(market_data=modified_market_data)

        bt_result1 = simulator.backtest(policy, start_time = rets.index[10],
            end_time = rets.index[20])

        # print(bt_result)
        # print(bt_result1)
        self.assertTrue(np.all(bt_result.w.isnull() == bt_result1.w.isnull()))
        self.assertTrue(np.allclose(bt_result.w, bt_result1.w, equal_nan=True))

    def test_backtest_with_ipos_and_delistings(self):
        """Test back-test with assets that both enter and exit."""
        rets = pd.DataFrame(self.returns.iloc[:, -10:], copy=True)
        volumes = pd.DataFrame(self.volumes.iloc[:, -9:], copy=True)
        prices = pd.DataFrame(self.prices.iloc[:, -9:], copy=True)
        rets.iloc[14:25, 1:3] = np.nan
        rets.iloc[9:17, 3:5] = np.nan
        rets.iloc[8:15, 5:7] = np.nan
        rets.iloc[16:29, 7:8] = np.nan
        # print(rets.iloc[10:20])

        modified_market_data = cvx.UserProvidedMarketData(
            returns=rets, volumes=volumes, prices=prices,
            cash_key='cash',
            min_history=pd.Timedelta('0d'))

        simulator = cvx.StockMarketSimulator(market_data=modified_market_data)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - 10 * cvx.FullCovariance(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)

        with self.assertRaises(ValueError):
            simulator.backtest(policy, start_time = rets.index[10],
            end_time = rets.index[9])

        bt_result = simulator.backtest(policy, start_time = rets.index[10],
            end_time = rets.index[20])

        # print(bt_result.w)

        self.assertTrue(set(bt_result.w.columns) == set(rets.columns))
        self.assertTrue(
            np.all(bt_result.w.iloc[:-1].isnull() == rets.iloc[
                10:20].isnull()))

    def test_backtest_with_difficult_universe_changes(self):
        """Test back-test with assets that both enter and exit at same time."""

        market_data, t_start, t_end = self._difficult_market_data()
        simulator = cvx.StockMarketSimulator(market_data=market_data)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - 10 * cvx.FullSigma(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)

        bt_result = simulator.run_backtest(
            policy, start_time = t_start, end_time = t_end)

        # print(bt_result.w)

        self.assertTrue(
            set(bt_result.w.columns) == set(market_data.full_universe))
        self.assertTrue(
            np.all(bt_result.w.iloc[:-1].isnull() == market_data.returns.iloc[
                10:20].isnull()))

    def test_result(self):
        """Test methods and properties of result."""
        sim = cvx.MarketSimulator(
            market_data = self.market_data, base_location=self.datadir)
        result = sim.backtest(cvx.Uniform(), pd.Timestamp(
            '2014-05-01'))
        result.plot(show=False)
        result.times_plot(show=False)
        str(result)
        for attribute in dir(result):
            # print(attribute, getattr(result, attribute))
            str(getattr(result, attribute))

    @staticmethod
    def _equal_logs(log1, log2, strip_pid=False):
        """Because first ~25 chars are datetime, next 20 are process info.

        Also need to skip first line if market data is doing masking of the
        df's, might have to change if we change logging logic there.
        """
        # print(log1)
        # print(log2)
        log1 = log1.split('\n')
        log2 = log2.split('\n')
        if 'Masking internal' in log1[0]:
            log1 = log1[1:] # pragma: no cover
        if 'Masking internal' in log2[0]:
            log2 = log2[1:]
        return [
            el[50 if strip_pid else 25:] for el in log1] == [
                el[50 if strip_pid else 25:] for el in log2]

    def test_logs(self):
        """Test correct recording of logs by BacktestResult."""

        sim = cvx.MarketSimulator(
            market_data = self.market_data, base_location=self.datadir)
        result = sim.backtest(cvx.Uniform(), pd.Timestamp(
            '2014-05-01'))
        result_base = result.logs
        self.assertGreater(len(result_base), 100)
        self.assertGreater(len(result_base.split('\n')), 10)

        opt_pol = cvx.SinglePeriodOptimization(
                cvx.ReturnsForecast(), [cvx.LongOnly(applies_to_cash=True)],
                solver=self.default_socp_solver) # OSQP may have randomness
        result = sim.backtest(opt_pol, pd.Timestamp('2014-05-01'))
        result_base1 = result.logs

        self.assertGreater(len(result_base1), len(result_base))
        self.assertGreater(len(result_base1.split('\n')), 10)

        # setting different root logger levels
        logger = logging.getLogger()
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger.setLevel(level)
            result = sim.backtest(cvx.Uniform(), pd.Timestamp(
                '2014-05-01'))
            self.assertTrue(self._equal_logs(result.logs, result_base))
            # check final level is correct
            self.assertEqual(logging.getLevelName(logger.level), level)

        # with multiprocessing
        policies = [cvx.Uniform(), opt_pol]

        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger.setLevel(level)
            result1, result2 = sim.backtest_many(
                policies, pd.Timestamp('2014-05-01'), parallel=True)
            self.assertTrue(
                self._equal_logs(result1.logs, result_base, strip_pid=True))
            self.assertTrue(
                self._equal_logs(result2.logs, result_base1, strip_pid=True))
            self.assertEqual(logging.getLevelName(logger.level), level)


if __name__ == '__main__':

    unittest.main(warnings='error') # pragma: no cover
