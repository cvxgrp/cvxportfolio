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
"""Unit tests for the market simulator and its backtest methods."""

import copy
import multiprocessing
import time
import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.errors import ConvexityError, ConvexSpecificationError
from cvxportfolio.simulator import MarketSimulator, StockMarketSimulator
from cvxportfolio.tests import CvxportfolioTest


class TestSimulator(CvxportfolioTest):
    """Test MarketSimulator and assorted end-to-end tests."""

    @classmethod
    def setUpClass(cls):
        """Also define some MarketData objects to share.

        We could probably refactor them and/re replace them with
        UserProvided ones. It's not so bad because each name is downloaded
        only once (thanks to `cls.data_grace_period`).
        """
        super(TestSimulator, cls).setUpClass()

        cls.market_data_1 = cvx.DownloadedMarketData(
            ['ZM', 'META'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_2 = cvx.DownloadedMarketData(
            ['ZM', 'META', 'AAPL'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_3 = cvx.DownloadedMarketData(
            ['META', 'AAPL'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_4 = cvx.DownloadedMarketData(
            ['AAPL', 'MSFT'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_5 = cvx.DownloadedMarketData(
            ['AAPL', 'ZM'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_6 = cvx.DownloadedMarketData(
            ['AAPL', 'MSFT', 'GE', 'ZM', 'META'],
            grace_period=cls.data_grace_period,
            base_location=cls.datadir)

        cls.market_data_7 = cvx.DownloadedMarketData(
            ['AAPL', 'MSFT', 'GE', 'ZM', 'META'],
            base_location=cls.datadir,
            grace_period=cls.data_grace_period,
            trading_frequency='monthly')

    def test_simulator_raises(self):
        """Test syntax checker of MarketSimulator."""

        with self.assertRaises(SyntaxError):
            MarketSimulator()

        with self.assertRaises(SyntaxError):
            StockMarketSimulator(returns=pd.DataFrame(
                [[0.]], index=[pd.Timestamp.today()], columns=['USDOLLAR']))

        with self.assertRaises(SyntaxError):
            MarketSimulator(volumes=pd.DataFrame(
                [[0.]], index=[pd.Timestamp.today()]))

        with self.assertRaises(SyntaxError):
            MarketSimulator(returns=pd.DataFrame(
                [[0.]], columns=['USDOLLAR'], index=[pd.Timestamp.today()]),
                volumes=pd.DataFrame([[0.]]),
                min_history=pd.Timedelta('0d'))

        # not raises
        _ = MarketSimulator(
            returns=pd.DataFrame([[0., 0.]], columns=['A', 'USDOLLAR'],
              index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')]),
            volumes=pd.DataFrame(
            [[0.]], columns=['A']), round_trades=False,
            min_history = pd.Timedelta('0d'))

        with self.assertRaises(SyntaxError):
            MarketSimulator(returns=pd.DataFrame(
                [[0., 0.]], index=[pd.Timestamp.today()],
                columns=['X', 'USDOLLAR']),
                volumes=pd.DataFrame([[0.]]),
                min_history = pd.Timedelta('0d'))

    def test_backtest_with_ipos_and_delistings(self):
        """Test back-test with assets that both enter and exit."""
        rets = pd.DataFrame(self.returns.iloc[:, -10:], copy=True)
        volumes = pd.DataFrame(self.volumes.iloc[:, -9:], copy=True)
        prices = pd.DataFrame(self.prices.iloc[:, -9:], copy=True)
        rets.iloc[14:25, 1:3] = np.nan
        rets.iloc[9:17, 3:5] = np.nan
        rets.iloc[8:15, 5:7] = np.nan
        rets.iloc[16:29, 7:8] = np.nan
        print(rets.iloc[10:20])

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

        print(bt_result.w)

        self.assertTrue(set(bt_result.w.columns) == set(rets.columns))
        self.assertTrue(
            np.all(bt_result.w.iloc[:-1].isnull() == rets.iloc[
                10:20].isnull()))

    def test_backtest_with_difficult_universe_changes(self):
        """Test back-test with assets that both enter and exit at same time."""
        rets = pd.DataFrame(self.returns.iloc[:, -10:], copy=True)
        volumes = pd.DataFrame(self.volumes.iloc[:, -9:], copy=True)
        prices = pd.DataFrame(self.prices.iloc[:, -9:], copy=True)
        rets.iloc[15:25, 1:3] = np.nan
        rets.iloc[9:17, 3:5] = np.nan
        rets.iloc[8:15, 5:7] = np.nan
        rets.iloc[17:29, 7:8] = np.nan
        print(rets.iloc[10:20])

        modified_market_data = cvx.UserProvidedMarketData(
            returns=rets, volumes=volumes, prices=prices,
            cash_key='cash',
            min_history=pd.Timedelta('0d'))

        simulator = cvx.StockMarketSimulator(market_data=modified_market_data)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - 10 * cvx.FullSigma(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)

        bt_result = simulator.run_backtest(policy, start_time = rets.index[10],
            end_time = rets.index[20])

        print(bt_result.w)

        self.assertTrue(set(bt_result.w.columns) == set(rets.columns))
        self.assertTrue(
            np.all(bt_result.w.iloc[:-1].isnull() == rets.iloc[
                10:20].isnull()))

    def test_prepare_data(self):
        """Test that (Downloaded)MarketData is created correctly."""
        market_data = cvx.DownloadedMarketData(
            ['ZM', 'META'], grace_period = self.data_grace_period,
            base_location=self.datadir)
        self.assertTrue(market_data.returns.shape[1] == 3)
        self.assertTrue(market_data.prices.shape[1] == 2)
        self.assertTrue(market_data.volumes.shape[1] == 2)
        # self.assertTrue( simulator.sigma_estimate.data.shape[1] == 2)
        self.assertTrue(np.isnan(market_data.returns.iloc[-1, 0]))
        self.assertTrue(np.isnan(market_data.volumes.iloc[-1, 1]))
        self.assertTrue(not np.isnan(market_data.prices.iloc[-1, 0]))
        self.assertTrue(
            market_data.returns.index[-1] == market_data.volumes.index[-1])
        self.assertTrue(
            market_data.returns.index[-1] == market_data.prices.index[-1])

    def test_holding_cost(self):
        """Test the simulator interface of cvx.HoldingCost."""

        t = self.returns.index[-20]

        # stock holding cost
        for i in range(10):
            np.random.seed(i)
            h_plus = np.random.randn(4)*10000
            h_plus[3] = 10000 - sum(h_plus[:-1])
            h_plus = pd.Series(h_plus)

            dividends = np.random.uniform(size=len(h_plus)-1) * 1E-4

            hcost = cvx.HoldingCost(short_fees=5, dividends=dividends)

            sim_hcost = hcost.simulate(
                t=t, h_plus=h_plus,
                t_next=t + pd.Timedelta('1d'))

            hcost = -(np.exp(np.log(1.05)/365.24)-1) * sum(
                -np.minimum(h_plus, 0.)[:-1])
            hcost += dividends @ h_plus[:-1]
            print(hcost, -sim_hcost)
            self.assertTrue(np.isclose(hcost, -sim_hcost))

    def test_transaction_cost_syntax(self):
        """Test syntax checks of (Stocks)TransactionCost."""

        t = self.returns.index[-20]

        past_returns, current_returns, past_volumes, current_volumes, \
            current_prices = self.market_data.serve(t)

        u = pd.Series(np.ones(len(current_prices)+1), self.universe)

        tcost = cvx.StocksTransactionCost()
        # syntax checks
        with self.assertRaises(SyntaxError):
            tcost.simulate(t, u=u,
                            past_returns=past_returns,
                            current_returns=current_returns,
                            past_volumes=past_volumes,
                            current_volumes=current_volumes,
                            current_prices=None)

        tcost = cvx.TransactionCost(pershare_cost=None,)
        tcost.simulate(t, u=u, current_prices=None,
                        past_returns=past_returns,
                        current_returns=current_returns,
                        past_volumes=past_volumes,
                        current_volumes=current_volumes)

        tcost = cvx.TransactionCost()
        with self.assertRaises(SyntaxError):
            tcost.simulate(t, u=u, current_prices=current_prices,
                            past_returns=past_returns,
                            current_returns=current_returns,
                            past_volumes=None,
                            current_volumes=None)

        tcost = cvx.TransactionCost(b=None)
        tcost.simulate(t, u=u, current_prices=current_prices,
                        past_returns=past_returns,
                        current_returns=current_returns,
                        past_volumes=None,
                        current_volumes=None)

    def test_transaction_cost(self):
        """Test (Stock)TransactionCost simulator's interface."""

        t = self.returns.index[-5]

        past_returns, current_returns, past_volumes, current_volumes, \
            current_prices = self.market_data.serve(t)

        # print(current_prices)

        n = len(current_prices)

        for i in range(10):
            np.random.seed(i)
            spreads = np.random.uniform(size=n)*1E-3
            u = np.random.uniform(size=n+1)*1E4
            u[-1] = -sum(u[:-1])
            u = pd.Series(u, self.universe)

            # pylint: disable=protected-access
            u = MarketSimulator._round_trade_vector(u, current_prices)

            tcost = cvx.StocksTransactionCost(a=spreads/2)

            sim_cost = tcost.simulate(
                t, u=u, current_prices=current_prices,
                past_returns=past_returns,
                current_returns=current_returns,
                past_volumes=past_volumes,
                current_volumes=current_volumes)

            shares = sum(np.abs(u[:-1] / current_prices))
            tcost = -0.005 * shares
            # print(tcost, sim_cost)
            tcost -= np.abs(u.iloc[:-1]) @ spreads / 2
            tcost -= sum((np.abs(u.iloc[:-1])**1.5
                ) * self.returns.loc[self.returns.index <=
                t].iloc[-252:, :-1].std(ddof=0) / np.sqrt(self.volumes.loc[t]))
            print(tcost, sim_cost)
            self.assertTrue(np.isclose(tcost, -sim_cost))

    def test_methods(self):
        """Test some methods of MarketSimulator."""
        simulator = MarketSimulator(
            # because we modify it
            market_data=copy.deepcopy(self.market_data_2),
            base_location=self.datadir)

        self.strip_tz_and_hour(simulator.market_data)

        for t in [pd.Timestamp('2023-04-13')]:

            # round trade

            for i in range(10):
                np.random.seed(i)
                tmp = np.random.uniform(size=4)*1000
                tmp[3] = -sum(tmp[:3])
                u = pd.Series(tmp, simulator.market_data.full_universe)

                # pylint: disable=protected-access
                rounded = simulator._round_trade_vector(
                    u, simulator.market_data.prices.loc[t])
                self.assertTrue(sum(rounded) == 0)
                self.assertTrue(
                    np.linalg.norm(rounded[:-1] - u[:-1])
                    < np.linalg.norm(simulator.market_data.prices.loc[t]/2))

                print(u)

    def test_simulate_policy(self):
        """Test basic policy simulation."""
        simulator = StockMarketSimulator(
            # because we modify it
            market_data=copy.deepcopy(self.market_data_3),
            base_location=self.datadir)

        # to fix this test
        self.strip_tz_and_hour(simulator.market_data)

        start_time = '2023-03-10'
        end_time = '2023-04-20'

        # hold
        policy = cvx.Hold()
        for i in range(10):
            np.random.seed(i)
            h = np.random.randn(3)*10000
            h[-1] = 10000 - sum(h[:-1])
            h0 = pd.Series(h, simulator.market_data.full_universe)
            h = pd.Series(h0, copy=True)

            policy.initialize_estimator_recursive(
                universe=simulator.market_data.full_universe,
                trading_calendar=simulator.market_data.trading_calendar(
                    start_time, end_time, include_end=False)
            )

            for (i, t) in enumerate(simulator.market_data.returns.index[
                    (simulator.market_data.returns.index >= start_time) & (
                    simulator.market_data.returns.index <= end_time)]):
                t_next = simulator.market_data.returns.index[i+1]
                oldcash = h.iloc[-1]
                past_returns, current_returns, past_volumes, current_volumes, \
                    current_prices = simulator.market_data.serve(t)
                h, _, _, costs, _ = simulator.simulate(
                    t=t, h=h, policy=policy, t_next=t_next,
                    past_returns=past_returns, current_returns=current_returns,
                    past_volumes=past_volumes, current_volumes=current_volumes,
                    current_prices=current_prices)
                tcost, hcost = costs['StocksTransactionCost'
                    ], costs['StocksHoldingCost']
                assert tcost == 0.
                # if np.all(h0[:2] > 0):
                #    assert hcost == 0.
                assert np.isclose(
                    (oldcash - hcost) * (1+simulator.market_data.returns.loc[
                        t, 'USDOLLAR']), h.iloc[-1])

            simh = h0[:-1] * simulator.market_data.prices.loc[pd.Timestamp(
                end_time) + pd.Timedelta('1d')
                ] / simulator.market_data.prices.loc[start_time]
            self.assertTrue(np.allclose(simh, h[:-1]))

        # proportional_trade
        policy = cvx.ProportionalTradeToTargets(
            targets=pd.DataFrame({pd.Timestamp(end_time)
                + pd.Timedelta('1d'):  pd.Series([0, 0, 1],
                simulator.market_data.returns.columns)}).T)

        for i in range(10):
            np.random.seed(i)
            h = np.random.randn(3)*10000
            h[-1] = 10000 - sum(h[:-1])
            h0 = pd.Series(h, simulator.market_data.returns.columns)
            h = pd.Series(h0, copy=True)
            policy.initialize_estimator_recursive(
                universe=simulator.market_data.full_universe,
                trading_calendar=simulator.market_data.trading_calendar(
                    start_time, end_time, include_end=False)
            )

            for i, t in enumerate(simulator.market_data.returns.index[
                    (simulator.market_data.returns.index >= start_time) &
                        (simulator.market_data.returns.index <= end_time)]):
                t_next = simulator.market_data.returns.index[i+1]
                oldcash = h.iloc[-1]
                past_returns, current_returns, past_volumes, current_volumes, \
                    current_prices = simulator.market_data.serve(t)
                h, _, _, costs, _ = simulator.simulate(
                    t=t, h=h, policy=policy, t_next=t_next,
                    past_returns=past_returns, current_returns=current_returns,
                    past_volumes=past_volumes, current_volumes=current_volumes,
                    current_prices=current_prices)
                tcost, hcost = costs['StocksTransactionCost'
                    ], costs['StocksHoldingCost']
                print(h)
                # print(tcost, stock_hcost, cash_hcost)

            self.assertTrue(
                np.all(np.abs(h[:-1])
                    < simulator.market_data.prices.loc[end_time]))

    def test_backtest(self):
        """Test simple back-test."""
        pol = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - cvx.ReturnsForecastError()
            - .5 * cvx.WorstCaseRisk(
                [cvx.FullCovariance(),
                cvx.DiagonalCovariance() + .25 * cvx.DiagonalCovariance()]),
            [cvx.LeverageLimit(1)], verbose=True,
            solver=self.default_socp_solver)
        sim = cvx.MarketSimulator(
            market_data=self.market_data_4, base_location=self.datadir)
        result = sim.backtest(pol, pd.Timestamp(
            '2023-01-01'), pd.Timestamp('2023-04-20'))

        print(result)

    def test_wrong_worstcase(self):
        """Test wrong worst-case convexity."""
        pol = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - cvx.ReturnsForecastError()
            - .5 * cvx.WorstCaseRisk(
                [-cvx.FullCovariance(),
                cvx.DiagonalCovariance() + .25 * cvx.DiagonalCovariance()]),
            [cvx.LeverageLimit(1)], verbose=True,
            solver=self.default_socp_solver)
        sim = cvx.MarketSimulator(
            market_data=self.market_data_4, base_location=self.datadir)

        with self.assertRaises(ConvexityError):
            sim.backtest(pol, pd.Timestamp(
                '2023-01-01'), pd.Timestamp('2023-04-20'))

    def test_backtest_changing_universe(self):
        """Test back-test with changing universe."""
        sim = cvx.MarketSimulator(
            market_data=self.market_data_5, base_location=self.datadir)
        pol = cvx.SinglePeriodOptimization(cvx.ReturnsForecast() -
                                           cvx.ReturnsForecastError() -
                                           .5 * cvx.FullCovariance(),
                                           [  # cvx.LongOnly(),
            cvx.LeverageLimit(1)], verbose=True, solver=self.default_qp_solver)

        result = sim.backtest(pol, pd.Timestamp(
            # zoom enters in mid-april
            '2020-04-01'), pd.Timestamp('2020-05-01'))

        ridx = result.w.index
        self.assertTrue(result.w['ZM'].isnull().sum() > 5)
        self.assertTrue(result.w['AAPL'].isnull().sum() < 2)
        self.assertTrue(len(ridx) == len(set(ridx)))
        self.assertTrue(len(ridx) == len(sim.market_data.returns.loc[
            (sim.market_data.returns.index >= ridx[0]) & (
            sim.market_data.returns.index <= ridx[-1])]))
        print(result)

    def test_multiple_backtest(self):
        """Test multiple back-tests (in parallel)."""
        pol = cvx.SinglePeriodOptimization(cvx.ReturnsForecast() -
                                           cvx.ReturnsForecastError() -
                                           .5 * cvx.FullCovariance(),
                                           [  # cvx.LongOnly(),
            cvx.LeverageLimit(1)], # verbose=True,
            solver=self.default_qp_solver)

        pol1 = cvx.Uniform()

        sim = cvx.MarketSimulator(
            market_data=self.market_data_4, base_location=self.datadir)

        with self.assertRaises(SyntaxError):
            sim.run_multiple_backtest([pol, pol1], pd.Timestamp(
                '2023-01-01'), pd.Timestamp('2023-04-20'), h=['hello'])

        with self.assertRaises(SyntaxError):
            sim.run_multiple_backtest(pol, pd.Timestamp(
                '2023-01-01'), pd.Timestamp('2023-04-20'), h=['hello'])

        result = sim.backtest(pol1, pd.Timestamp(
            '2023-01-01'), pd.Timestamp('2023-04-20'))

        result2, result3 = sim.backtest_many(
            [pol, pol1], pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-04-20'))

        self.assertTrue(np.all(result.h == result3.h))

        # with user-provided h
        good_h = pd.Series([0, 0, 1E6], index=['AAPL', 'MSFT', 'USDOLLAR'])
        result4, result5 = sim.backtest_many(
            [pol, pol1], start_time=pd.Timestamp('2023-01-01'),
            end_time=pd.Timestamp('2023-04-20'), h=[good_h, good_h])

        self.assertTrue(np.all(result2.h == result4.h))
        self.assertTrue(np.all(result3.h == result5.h))

        # shuffled h
        good_h_shuffled = good_h.iloc[::-1]
        result6, result7 = sim.backtest_many(
            [pol, pol1], start_time=pd.Timestamp('2023-01-01'),
            end_time=pd.Timestamp('2023-04-20'),
            h=[good_h_shuffled, good_h_shuffled])

        self.assertTrue(np.all(result2.h == result6.h))
        self.assertTrue(np.all(result3.h == result7.h))

        # bad h
        bad_h = pd.Series([0, 0, 1E6], index=['AAPL_bad', 'MSFT', 'USDOLLAR'])
        with self.assertRaises(ValueError):
            sim.backtest_many(
                [pol, pol1], start_time=pd.Timestamp('2023-01-01'),
                end_time=pd.Timestamp('2023-04-20'), h=[bad_h, good_h])

    def test_multiple_backtest2(self):
        """Test re-use of a worker process."""
        cpus = multiprocessing.cpu_count()

        sim = cvx.MarketSimulator(
            market_data=self.market_data_4, base_location=self.datadir)
        pols = [cvx.SinglePeriodOptimization(cvx.ReturnsForecast()
            - 1 * cvx.FullCovariance(), [cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)
                for i in range(cpus*2)]
        results = sim.backtest_many(pols, pd.Timestamp(
            '2023-01-01'), pd.Timestamp('2023-01-15'), parallel=True)
        sharpes = [result.sharpe_ratio for result in results]
        self.assertTrue(len(set(sharpes)) == 1)

    def test_multiple_backtest3(self):
        """Test benchmarks."""

        sim = cvx.MarketSimulator(
            market_data=self.market_data_4, base_location=self.datadir)
        pols = [
            cvx.SinglePeriodOptimization(cvx.ReturnsForecast(
            ) - 1 * cvx.FullCovariance(), [cvx.LeverageLimit(1)],
            solver=self.default_qp_solver),

            cvx.SinglePeriodOptimization(cvx.ReturnsForecast(
            ) - 1 * cvx.FullCovariance(), [cvx.LeverageLimit(1)],
                benchmark=cvx.Uniform, solver=self.default_qp_solver),

            cvx.SinglePeriodOptimization(cvx.ReturnsForecast(
            ) - 1 * cvx.FullCovariance(), [cvx.LeverageLimit(1)],
                benchmark=cvx.MarketBenchmark, solver=self.default_qp_solver),
        ]
        results = sim.backtest_many(pols, pd.Timestamp(
            '2023-01-01'), pd.Timestamp('2023-01-15'), parallel=True)
        print(np.linalg.norm(results[0].w.sum()[:2] - .5))
        print(np.linalg.norm(results[1].w.sum()[:2] - .5))
        print(np.linalg.norm(results[2].w.sum()[:2] - .5))
        self.assertTrue(np.linalg.norm(results[1].w.sum()[
                        :2] - .5) < np.linalg.norm(
                        results[0].w.sum()[:2] - .5))
        self.assertTrue(np.linalg.norm(results[1].w.sum()[
                        :2] - .5) < np.linalg.norm(
                        results[2].w.sum()[:2] - .5))

    def test_multiple_backtest4(self):
        """Test _downsample and offline cache."""

        time_first = 0.
        results_first = []
        for downsampling in ['weekly', 'monthly', 'quarterly', 'annual']:
            market_data = cvx.DownloadedMarketData(
                ['AAPL', 'MSFT', 'GE', 'ZM', 'META'],
                base_location=self.datadir,
                grace_period=self.data_grace_period,
                trading_frequency=downsampling)
            sim = cvx.MarketSimulator(
                market_data=market_data,
                base_location=self.datadir)
            pol = cvx.SinglePeriodOptimization(cvx.ReturnsForecast(
            ) - 1 * cvx.FullCovariance() - cvx.TransactionCost(exponent=1.5),
            [cvx.LeverageLimit(1)], solver=self.default_socp_solver)
            s = time.time()
            results_first.append(sim.backtest(pol, pd.Timestamp('2020-12-01')))
            print(results_first[-1])
            time_first += time.time() - s

        with self.assertRaises(SyntaxError):
            _ = cvx.DownloadedMarketData(
                ['AAPL', 'MSFT', 'GE', 'ZM', 'META'],
                base_location=self.datadir,
                grace_period=self.data_grace_period,
                trading_frequency='unsupported')

        time_second = 0.
        results_second = []
        for downsampling in ['weekly', 'monthly', 'quarterly', 'annual']:
            market_data = cvx.DownloadedMarketData(
                ['AAPL', 'MSFT', 'GE', 'ZM', 'META'],
                base_location=self.datadir,
                grace_period=self.data_grace_period,
                trading_frequency=downsampling)
            sim = cvx.MarketSimulator(
                market_data=market_data,
                base_location=self.datadir)
            pol = cvx.SinglePeriodOptimization(cvx.ReturnsForecast(
            ) - 1 * cvx.FullCovariance() - cvx.TransactionCost(exponent=1.5),
            [cvx.LeverageLimit(1)],
            solver=self.default_socp_solver)
            s = time.time()
            results_second.append(sim.backtest(
                pol, pd.Timestamp('2020-12-01')))
            print(results_second[-1])
            time_second += time.time() - s

        # example is too small to see speed difference w/ cache
        # sadly we have to drop this test element
        # self.assertTrue(time_second < time_first)
        print(time_second, time_first)
        for i, _ in enumerate(results_first):
            self.assertTrue(
                np.isclose(_.sharpe_ratio, results_second[i].sharpe_ratio))

    def test_result(self):
        """Test methods and properties of result."""
        sim = cvx.MarketSimulator(
            market_data = self.market_data_6, base_location=self.datadir)
        result = sim.backtest(cvx.Uniform(), pd.Timestamp(
            '2023-01-01'))
        result.plot(show=False)
        print(result)
        for attribute in dir(result):
            print(attribute, getattr(result, attribute))

    def test_spo_benchmark(self):
        """Test the effect of benchmark on SPO policies."""

        sim = cvx.MarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        objective = cvx.ReturnsForecast() - 10 * cvx.FullCovariance()
        constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]

        myunif = pd.Series(0.2, ['AAPL', 'MSFT', 'GE', 'ZM', 'META'])
        myunif['USDOLLAR'] = 0.

        policies = [
            cvx.SinglePeriodOptimization(
                objective, constraints, benchmark=bm,
                solver=self.default_qp_solver)
            for bm in
                [cvx.AllCash(), cvx.Uniform(), cvx.MarketBenchmark(), myunif]]

        results = sim.backtest_many(
            policies, start_time='2023-01-01',
            parallel=False)  # important for test coverage!!

        # check myunif is the same as uniform
        self.assertTrue(np.isclose(
            results[1].sharpe_ratio, results[3].sharpe_ratio))

        # check cash benchmark sol has higher cash weights
        self.assertTrue(results[0].w.USDOLLAR.mean() >=
                        results[1].w.USDOLLAR.mean())
        self.assertTrue(results[0].w.USDOLLAR.mean() >=
                        results[2].w.USDOLLAR.mean())

        # check that uniform bm sol is closer
        # to uniform alloc than market bm sol
        norm_smaller = (
            (results[1].w.iloc[:, :-1] - 0.2)
              ** 2).mean(1) < ((results[2].w.iloc[:, :-1] - 0.2)**2).mean(1)
        print(norm_smaller.describe())
        self.assertTrue(norm_smaller.mean() > .5)

    def test_market_neutral(self):
        """Test SPO with market neutral constraint."""
        market_data = cvx.DownloadedMarketData(
            ['AAPL', 'MSFT', 'GE', 'GOOG', 'META', 'GLD'],
            base_location=self.datadir,
            grace_period=self.data_grace_period,
            trading_frequency='monthly')
        sim = cvx.MarketSimulator(
            market_data=market_data, base_location=self.datadir)

        objective = cvx.ReturnsForecast() - 2 * cvx.FullCovariance()

        policies = [cvx.SinglePeriodOptimization(objective, co,
            solver=self.default_qp_solver) for co in [
            [], [cvx.MarketNeutral()], [cvx.DollarNeutral()]]]

        results = sim.backtest_many(
            policies, start_time='2023-01-01',
            parallel=False)  # important for test coverage

        print(results)

        # check that market neutral sol is closer to
        dists_from_dollar_neutral = [
            np.abs(result.w.iloc[:, -1] - 1).mean() for result in results]
        print('dists_from_dollar_neutral')
        print(dists_from_dollar_neutral)
        self.assertTrue(
            dists_from_dollar_neutral[2] < dists_from_dollar_neutral[1])
        self.assertTrue(
            dists_from_dollar_neutral[1] < dists_from_dollar_neutral[0])

    def test_timed_constraints(self):
        """Test some constraints that depend on time."""

        sim = cvx.StockMarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        # cvx.NoTrade
        objective = cvx.ReturnsForecast() - 2 * cvx.FullCovariance()

        no_trade_ts = [sim.market_data.returns.index[-2],
                       sim.market_data.returns.index[-6]]

        policy = cvx.SinglePeriodOptimization(
            objective, [cvx.NoTrade('AAPL', no_trade_ts)],
            solver=self.default_qp_solver)

        result = sim.backtest(policy, start_time='2023-01-01')
        print(result.z)
        for t in no_trade_ts:
            self.assertTrue(np.isclose(result.z['AAPL'].loc[t], 0., atol=1E-3))

        # cvx.MinWeightsAtTimes, cvx.MaxWeightsAtTimes
        policies = [cvx.MultiPeriodOpt(
            objective - cvx.StocksTransactionCost(),
            [cvx.MinWeightsAtTimes(0., no_trade_ts),
            cvx.MaxWeightsAtTimes(0., no_trade_ts)],
            planning_horizon=p,
            solver=self.default_socp_solver) for p in [1, 3, 5]]

        results = sim.backtest_many(
            policies, start_time='2023-01-01', initial_value=1E6,
            parallel=False)  # important for test coverage
        print(results)

        total_tcosts = [result.costs['StocksTransactionCost'].sum()
                        for result in results]
        print(total_tcosts)
        self.assertTrue(total_tcosts[0] > total_tcosts[1])
        self.assertTrue(total_tcosts[1] > total_tcosts[2])

    def test_eq_soft_constraints(self):
        """We check that soft DollarNeutral penalizes non-dollar-neutrality."""

        sim = cvx.MarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        objective = cvx.ReturnsForecast() - 5 * cvx.FullCovariance()

        policies = [cvx.SinglePeriodOptimization(
            objective - cvx.SoftConstraint(cvx.DollarNeutral()) * gamma)
            for gamma in [.0001, .001, .01]]
        policies.append(cvx.SinglePeriodOptimization(
            objective, [cvx.DollarNeutral()]))
        results = sim.backtest_many(
            policies, start_time='2023-01-01',
            parallel=False)  # important for test coverage
        print(results)
        allcashpos = [((res.w.iloc[:, -1]-1)**2).mean() for res in results]
        print(allcashpos)
        self.assertTrue(allcashpos[0] > allcashpos[1])
        self.assertTrue(allcashpos[1] > allcashpos[2])
        self.assertTrue(allcashpos[2] > allcashpos[3])

    def test_ineq_soft_constraints(self):
        """We check that soft LongOnly penalizes shorts."""

        sim = cvx.MarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        objective = cvx.ReturnsForecast() - .5 * cvx.FullCovariance()

        policies = [cvx.SinglePeriodOptimization(
            objective - cvx.SoftConstraint(cvx.LongOnly()) * gamma,
            [cvx.MarketNeutral()]) for gamma in [.0001, .001, .01]]
        policies.append(cvx.SinglePeriodOptimization(
            objective, [cvx.LongOnly(), cvx.MarketNeutral()]))
        results = sim.backtest_many(
            policies, start_time='2023-01-01',
            parallel=False)  # important for test coverage
        print(results)
        allshorts = [np.minimum(res.w.iloc[:, :-1], 0.).sum().sum()
                     for res in results]
        print(allshorts)
        self.assertTrue(allshorts[0] < allshorts[1])
        self.assertTrue(allshorts[1] < allshorts[2])
        self.assertTrue(allshorts[2] < allshorts[3])

    def test_cost_constraints(self):
        """We check that cost constraints work as expected."""

        sim = cvx.MarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        policies = [
            cvx.SinglePeriodOptimization(
                cvx.ReturnsForecast(), [cvx.FullCovariance() <= el**2],
                solver=self.default_socp_solver)
            for el in [0.01, .02, .05, .1]]

        results = sim.backtest_many(
            policies, start_time='2023-01-01',
            parallel=False)  # important for test coverage

        print(results)

        self.assertTrue(results[0].volatility < results[1].volatility)
        self.assertTrue(results[1].volatility < results[2].volatility)
        self.assertTrue(results[2].volatility < results[3].volatility)

    def test_dcp_convex_raises(self):
        """Test that some errors are thrown at wrong problem specifications."""

        market_data = cvx.DownloadedMarketData(
            ['AAPL'],
            base_location=self.datadir,
            grace_period=self.data_grace_period)
        sim = cvx.MarketSimulator(
            market_data=market_data, base_location=self.datadir)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast(), [cvx.FullCovariance() >= 2])

        with self.assertRaises(ConvexSpecificationError):
            sim.backtest(policy)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() + .5 * cvx.FullCovariance())

        with self.assertRaises(ConvexityError):
            sim.backtest(policy)

    def test_hyperparameters_optimize(self):
        """Test hyperparameter optimization."""

        gamma_risk = cvx.Gamma()
        gamma_trade = cvx.Gamma()
        objective = cvx.ReturnsForecast() - gamma_risk * cvx.FullCovariance()\
             - gamma_trade * cvx.StocksTransactionCost()
        policy = cvx.SinglePeriodOptimization(
            objective, [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_socp_solver)

        simulator = cvx.StockMarketSimulator(
            market_data=self.market_data_7, base_location=self.datadir)

        self.assertTrue(gamma_risk.current_value == 1.)
        self.assertTrue(gamma_trade.current_value == 1.)

        simulator.optimize_hyperparameters(
            policy, start_time='2023-01-01', end_time='2023-10-01')

        self.assertTrue(np.isclose(gamma_risk.current_value, 1.4641))
        self.assertTrue(np.isclose(gamma_trade.current_value, 0.385543289))

    def test_cancel_trades(self):
        """Test trade cancellation."""

        market_data = cvx.DownloadedMarketData(
            ['AAPL', 'ZM'],
            base_location=self.datadir,
            grace_period=self.data_grace_period,
            trading_frequency='monthly')
        sim = cvx.MarketSimulator(
            market_data=market_data, base_location=self.datadir)

        sim.market_data.volumes['ZM'] = 0.

        objective = cvx.ReturnsForecast() - 5 * cvx.FullCovariance()
        policy = cvx.SinglePeriodOptimization(
            objective, [cvx.LongOnly(), cvx.LeverageLimit(1)])

        sim.backtest(policy, start_time='2023-01-01')

    def test_svd_covariance_forecaster(self):
        """Test SVD covariance forecaster in simulation."""

        market_data = cvx.DownloadedMarketData(
            ['AAPL', 'MSFT', 'GE', 'ZM', 'META', 'GOOG', 'GLD'],
            base_location=self.datadir,
            grace_period=self.data_grace_period,
            trading_frequency='quarterly')
        sim = cvx.StockMarketSimulator(
            market_data=market_data, base_location=self.datadir)

        objective = cvx.ReturnsForecast() - 5 * cvx.FactorModel(
            num_factors=2, Sigma=None)
        policy = cvx.SinglePeriodOpt(
            objective, [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)

        result_svd = sim.backtest(policy, start_time='2020-01-01',
            end_time='2023-09-01')

        objective = cvx.ReturnsForecast() - 5 * cvx.FactorModelCovariance(
            num_factors=2)
        policy = cvx.SinglePeriodOptimization(
            objective, [cvx.LongOnly(), cvx.LeverageLimit(1)],
            solver=self.default_qp_solver)

        result_eig = sim.backtest(policy, start_time='2020-01-01',
            end_time='2023-09-01')

        self.assertTrue(result_svd.sharpe_ratio > result_eig.sharpe_ratio)

        print(result_svd)
        print(result_eig)

    def test_bankruptcy(self):
        """Test policy bankruptcy."""

        market_data = cvx.DownloadedMarketData(
            ['SPY', 'QQQ'],
            base_location=self.datadir,
            grace_period=self.data_grace_period)
        sim = cvx.StockMarketSimulator(
            market_data=market_data, base_location=self.datadir)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast(), [cvx.LeverageLimit(20)],
            solver=self.default_qp_solver)
        with self.assertLogs(level='WARNING') as _:
            result = sim.backtest(policy,
                start_time='2020-02-15', end_time='2020-04-15')
        # print(result)
        print(result.h)
        self.assertTrue(result.h.shape[0] < 20)
        self.assertTrue(result.final_value < 0)

    def test_cache_missing_signature(self):
        """Test backtest with missing market data signature."""
        md = cvx.UserProvidedMarketData(
            returns=self.returns, volumes=self.volumes,
            cash_key='cash', base_location=self.datadir,
            min_history=pd.Timedelta('0d'))
        md.partial_universe_signature = lambda x: None

        simulator = cvx.MarketSimulator(market_data=md)

        # print(os.listdir(self.datadir/'cache'))

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - .5 * cvx.FullCovariance(),
            [cvx.LongOnly(applies_to_cash=True)])

        simulator.backtest(
            policy, start_time = self.returns.index[10],
            end_time = self.returns.index[20],
            )

if __name__ == '__main__':

    unittest.main(warnings='error') # pragma: no cover
