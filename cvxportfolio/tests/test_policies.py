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
"""Unit tests for the policy objects."""

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.errors import DataError, PortfolioOptimizationError
from cvxportfolio.forecast import HistoricalFactorizedCovariance
from cvxportfolio.tests import CvxportfolioTest

VALUES_IN_TIME_DUMMY_KWARGS = {
    'past_returns': None,
    'current_prices': None,
    'past_volumes': None
}

class TestPolicies(CvxportfolioTest):
    """Test trading policies."""

    def test_hold(self):
        """Test hold policy."""
        hold = cvx.Hold()
        w = pd.Series(0.5, ["AAPL", "CASH"])
        self.assertTrue(np.all(
            hold.values_in_time_recursive(current_weights=w).values
            == w.values))

    def test_rank_and_long_short(self):
        """Test rank-and-long-short policy."""
        w = pd.Series(0.25, ["AAPL", "TSLA", "GOOGL", "CASH"])
        signal = pd.Series([1, 2, 3], ["AAPL", "TSLA", "GOOGL"])
        num_long = 1
        num_short = 1
        target_leverage = 3.0
        rls = cvx.RankAndLongShort(
            signal=signal,
            num_long=num_long,
            num_short=num_short,
            target_leverage=target_leverage,
        )
        wplus = rls.values_in_time_recursive(t=None, current_weights=w)

        self.assertTrue(wplus["CASH"] == 1)
        self.assertTrue(wplus["TSLA"] == 0)
        self.assertTrue(wplus["AAPL"] == -wplus["GOOGL"])
        self.assertTrue(np.abs(wplus[:-1]).sum() == 3)

        index = pd.date_range("2020-01-01", "2020-01-03")
        signal = pd.DataFrame(
            {
                "AAPL": pd.Series([1, 1.9, 3], index),
                "TSLA": pd.Series([3, 2.1, 1], index),
                "GOOGL": pd.Series([4, 4, 4], index),
            }
        )
        rls = cvx.RankAndLongShort(
            signal=signal,
            num_long=num_long,
            num_short=num_short,
            target_leverage=target_leverage,
        )
        wplus = rls.values_in_time_recursive(t=index[0], current_weights=w)

        self.assertTrue(wplus["CASH"] == 1)
        self.assertTrue(wplus["TSLA"] == 0)
        self.assertTrue(wplus["AAPL"] == -wplus["GOOGL"])
        self.assertTrue(np.abs(wplus[:-1]).sum() == 3)
        wplus = rls.values_in_time_recursive(t=index[1], current_weights=w)

        self.assertTrue(wplus["CASH"] == 1)
        self.assertTrue(wplus["TSLA"] == 0)
        self.assertTrue(wplus["AAPL"] == -wplus["GOOGL"])
        self.assertTrue(np.abs(wplus[:-1]).sum() == 3)
        wplus = rls.values_in_time_recursive(t=index[2], current_weights=w)

        self.assertTrue(wplus["CASH"] == 1)
        self.assertTrue(wplus["AAPL"] == 0)
        self.assertTrue(wplus["TSLA"] == -wplus["GOOGL"])
        self.assertTrue(np.abs(wplus[:-1]).sum() == 3)

    def test_proportional_trade(self):
        """Test proportional trade policy."""

        a = pd.Series(1., self.returns.columns)
        a.iloc[-1] = 1 - sum(a.iloc[:-1])
        b = pd.Series(-1., self.returns.columns)
        b.iloc[-1] = 1 - sum(b.iloc[:-1])

        targets = pd.DataFrame({self.returns.index[3]: a,
                                self.returns.index[15]: b
                                }).T
        policy = cvx.ProportionalTradeToTargets(targets)

        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        start_portfolio = pd.Series(
            np.random.randn(
                self.returns.shape[1]),
            self.returns.columns)
        start_portfolio.iloc[-1] = 1 - sum(start_portfolio.iloc[:-1])
        for t in self.returns.index[:17]:
            print(t)
            print(start_portfolio)

            wplus = policy.values_in_time_recursive(
                t=t, current_weights=start_portfolio)
            trade = wplus - start_portfolio.values
            start_portfolio = pd.Series(wplus, copy=True)

            if t in targets.index:
                self.assertTrue(np.all(start_portfolio == targets.loc[t]))
        print('trade', trade)
        self.assertTrue(np.allclose(trade, 0.))

    def test_sell_all(self):
        """Test sell-all policy."""
        start_portfolio = pd.Series(
            np.random.randn(
                self.returns.shape[1]),
            self.returns.columns)
        policy = cvx.SellAll()
        t = pd.Timestamp('2022-01-01')
        wplus = policy.values_in_time_recursive(
            t=t, past_returns=self.returns)
        allcash = np.zeros(len(start_portfolio))
        allcash[-1] = 1
        assert isinstance(wplus, pd.Series)
        assert np.allclose(allcash, wplus)

    def test_fixed_trade(self):
        """Test fixed trade policy."""
        fixed_trades = pd.DataFrame(
            np.random.randn(
                len(self.returns),
                self.returns.shape[1]),
            index=self.returns.index,
            columns=self.returns.columns)

        policy = cvx.FixedTrades(fixed_trades)
        t = self.returns.index[123]
        w = pd.Series(0., self.returns.columns)
        wplus = policy.values_in_time_recursive(
            t=t, current_weights=w,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.all(wplus-w == fixed_trades.loc[t]))
        w = wplus-w
        t = pd.Timestamp('1900-01-01')
        wplus = policy.values_in_time_recursive(
            t=t, current_weights=w,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.all(wplus-w == 0.))

    def test_fixed_weights(self):
        """Test fixed weights policy."""
        fixed_weights = pd.DataFrame(
            np.random.randn(
                len(self.returns),
                self.returns.shape[1]),
            index=self.returns.index,
            columns=self.returns.columns)

        policy = cvx.FixedWeights(fixed_weights)
        t = self.returns.index[123]
        wplus = policy.values_in_time_recursive(
            t=t, current_weights=pd.Series(0., self.returns.columns),
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.all(wplus == fixed_weights.loc[t]))

        t = self.returns.index[111]
        wplus = policy.values_in_time_recursive(
            t=t, current_weights=fixed_weights.iloc[110],
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS)
        self.assertTrue(np.allclose( wplus, fixed_weights.loc[t]))

        t = pd.Timestamp('1900-01-01')
        wplus1 = policy.values_in_time_recursive(t=t, current_weights=wplus,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.all(wplus1 == wplus))

    def test_periodic_rebalance(self):
        """Test periodic rebalance policy."""

        target = pd.Series(np.random.uniform(
            size=self.returns.shape[1]), self.returns.columns)
        target /= sum(target)
        rebalancing_times = pd.date_range(
            start=self.returns.index[0], end=self.returns.index[-1], freq='7d')

        policy = cvx.PeriodicRebalance(
            target, rebalancing_times=rebalancing_times)
        init = pd.Series(np.random.randn(
            self.returns.shape[1]), self.returns.columns)

        wplus = policy.values_in_time_recursive(
            t=rebalancing_times[0], current_weights=init,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.allclose(wplus, target))

        wplus = policy.values_in_time_recursive(
            t=rebalancing_times[0] + pd.Timedelta('1d'), current_weights=init,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.allclose(wplus, init))

    def test_uniform(self):
        """Test uniform allocation."""
        pol = cvx.Uniform()
        pol.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)

        init = pd.Series(np.random.randn(
            self.returns.shape[1]), self.returns.columns)
        wplus = pol.values_in_time_recursive(
            t=self.returns.index[123], current_weights=init,
            current_portfolio_value=1000,
            **VALUES_IN_TIME_DUMMY_KWARGS,
            )
        self.assertTrue(np.allclose(
            wplus[:-1],
            np.ones(self.returns.shape[1]-1)/(self.returns.shape[1]-1)))

    def test_proportional_rebalance(self):
        """Test the proportional rebalance policy."""

        target = pd.Series(np.random.uniform(
            size=self.returns.shape[1]), self.returns.columns)
        target /= sum(target)
        target_matching_times = self.returns.index[::3]

        policy = cvx.ProportionalRebalance(
            target, target_matching_times=target_matching_times)
        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)

        init = pd.Series(np.random.randn(
            self.returns.shape[1]), self.returns.columns)

        wplus = policy.values_in_time_recursive(
            t=self.returns.index[1], current_weights=init)
        trade = wplus - init
        wplus2 = policy.values_in_time_recursive(
            t=self.returns.index[2], current_weights=wplus)
        trade2 = wplus2 - wplus
        self.assertTrue(np.allclose(trade, trade2))
        print(trade + trade2 + init, target)
        self.assertTrue(np.allclose(trade + trade2 + init, target))

    def test_adaptive_rebalance(self):
        """Test adaptive rebalance policy."""
        np.random.seed(0)
        target = pd.Series(
            np.random.uniform(
                size=self.returns.shape[1]),
            self.returns.columns)
        target /= sum(target)
        target = pd.DataFrame({ind: target for ind in self.returns.index}).T

        init = pd.Series(np.random.uniform(
            size=self.returns.shape[1]), self.returns.columns)
        init /= sum(init)

        for tracking_error in [0.01, .02, .05, .1]:
            policy = cvx.AdaptiveRebalance(
                target, tracking_error=tracking_error)
            wplus = policy.values_in_time_recursive(
                t=self.returns.index[1], current_weights=init)
            self.assertTrue(np.allclose(wplus, target.iloc[0]))

        for tracking_error in [.2, .5]:
            policy = cvx.AdaptiveRebalance(
                target, tracking_error=tracking_error)
            wplus = policy.values_in_time_recursive(
                t=self.returns.index[1], current_weights=init)
            self.assertTrue(np.allclose(wplus - init, 0.))

    def test_single_period_optimization(self):
        """Test basic SPO."""

        return_forecast = cvx.ReturnsForecast()
        risk_forecast = cvx.FullCovariance(
            HistoricalFactorizedCovariance(kelly=False))
        tcost = cvx.TransactionCost(
            a=1E-3/2, b=None, exponent=2)

        policy = cvx.SinglePeriodOptimization(
            return_forecast
            - 2 * risk_forecast
            # - TcostModel(half_spread=5 * 1E-4)  # , power=2)
            - tcost,
            constraints=[
                cvx.LongOnly(applies_to_cash=False), cvx.LeverageLimit(1)],
            include_cash_return=False,
            # verbose=True,
            solver=self.default_socp_solver)

        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        # policy.compile_to_cvxpy()

        curw = np.zeros(self.N)
        curw[-1] = 1.

        result = policy.values_in_time_recursive(
            t=self.returns.index[121],
            current_weights=pd.Series(
                curw,
                self.returns.columns),
            current_portfolio_value=1000,
            past_returns=self.returns.iloc[:121],
            past_volumes=self.volumes.iloc[:121],
            current_prices=pd.Series(1., self.volumes.columns))

        cvxportfolio_result = pd.Series(result, self.returns.columns)

        print(cvxportfolio_result)

        # print(np.linalg.eigh(self.returns.iloc[:121, :-1].cov().values)[0])

        # REPLICATE WITH CVXPY

        covariance = self.returns.iloc[:121, :-1].cov(ddof=0).values
        w = cp.Variable(self.N)
        cp.Problem(
            cp.Maximize(w[:-1].T @ self.returns.iloc[:121, :-1].mean().values -
            2 * cp.quad_form(w[:-1], covariance) -
            5 * 1E-4 * cp.sum(cp.abs(w - curw)[:-1])),
            [w >= 0, w <= 1, sum(w) == 1]).solve(solver='ECOS')

        cvxpy_result = pd.Series(w.value, self.returns.columns)

        print(cvxpy_result)

        print(cvxportfolio_result - cvxpy_result)
        self.assertTrue(np.allclose(
            cvxportfolio_result - cvxpy_result, 0., atol=1e-5))

    def test_single_period_optimization_solve_twice(self):
        """Test resolve of SPO policy with Cvxpy parameters."""

        return_forecast = cvx.ReturnsForecast()
        risk_forecast = cvx.FullCovariance()

        policy = cvx.SinglePeriodOptimization(
            return_forecast
            - 2 * risk_forecast
            - cvx.TransactionCost(a=5 * 1E-4, b=0.),
            constraints=[cvx.LongOnly(), cvx.LeverageLimit(1)],
            # verbose=True,
            solver=self.default_socp_solver)

        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        # policy.compile_to_cvxpy()

        curw = np.zeros(self.N)
        curw[-1] = 1.

        result = policy.values_in_time_recursive(
            t=self.returns.index[134],
            current_weights=pd.Series(
                curw,
                self.returns.columns),
            current_portfolio_value=1000,
            past_returns=self.returns.iloc[:134],
            past_volumes=self.volumes.iloc[:134],
            current_prices=pd.Series(1., self.volumes.columns)) - curw

        self.assertFalse(np.allclose(result, 0.))

        curw += result

        result2 = policy.values_in_time_recursive(
            t=self.returns.index[134],
            current_weights=pd.Series(
                curw,
                self.returns.columns),
            current_portfolio_value=1000,
            past_returns=self.returns.iloc[:134],
            past_volumes=self.volumes.iloc[:134],
            current_prices=pd.Series(1., self.volumes.columns)) - curw

        self.assertTrue(np.allclose(result2, 0., atol=1e-7))

    def test_single_period_optimization_infeasible(self):
        """Test SPO policy with infeasible result."""

        return_forecast = cvx.ReturnsForecast()
        risk_forecast = cvx.FullCovariance()
        policy = cvx.SinglePeriodOptimization(
            return_forecast
            - 2 * risk_forecast
            - cvx.TransactionCost(a=5 * 1E-4, b=0.),
            constraints=[cvx.LongOnly(), cvx.LeverageLimit(1),
                cvx.MaxWeights(-1)],
            # verbose=True,
            solver=self.default_socp_solver)

        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        # policy.compile_to_cvxpy()

        curw = np.zeros(self.N)
        curw[-1] = 1.

        with self.assertRaises(PortfolioOptimizationError):
            policy.values_in_time_recursive(
                t=self.returns.index[134],
                current_weights=pd.Series(
                    curw,
                    self.returns.columns),
                current_portfolio_value=1000,
                past_returns=self.returns.iloc[:134],
                past_volumes=self.volumes.iloc[:134],
                current_prices=pd.Series(1., self.volumes.columns))

    def test_single_period_optimization_unbounded(self):
        """Test SPO policy with unbounded result."""

        return_forecast = cvx.ReturnsForecast()
        policy = cvx.SinglePeriodOptimization(
            return_forecast,
            constraints=[cvx.LongOnly(applies_to_cash=False)],
            # verbose=True,
            solver=self.default_socp_solver)

        policy.initialize_estimator_recursive(
            universe=self.returns.columns, trading_calendar=self.returns.index)
        # policy.compile_to_cvxpy()

        curw = np.zeros(self.N)
        curw[-1] = 1.

        with self.assertRaises(PortfolioOptimizationError):
            policy.values_in_time_recursive(
                t=self.returns.index[134],
                current_weights=pd.Series(
                    curw,
                    self.returns.columns),
                current_portfolio_value=1000,
                past_returns=self.returns.iloc[:134],
                past_volumes=self.volumes.iloc[:134],
                current_prices=pd.Series(1., self.volumes.columns))

    def test_multi_period_optimization2(self):
        """Test that MPO1 and MPO2 and MPO5 return same if no tcost, and diff.

        if tcost.
        """

        results = []
        for planning_horizon in [1, 2, 5]:
            return_forecast = cvx.ReturnsForecast()
            risk_forecast = cvx.FullCovariance()
            policy = cvx.MultiPeriodOptimization(
                return_forecast
                # - TcostModel(half_spread=5 * 1E-4)  # , power=2)
                - 10 * risk_forecast,
                constraints=[cvx.LongOnly(), cvx.LeverageLimit(1)],
                # verbose=True,
                planning_horizon=planning_horizon,
                solver=self.default_socp_solver)

            policy.initialize_estimator_recursive(
                universe=self.returns.columns,
                trading_calendar=self.returns.index)
            # policy.compile_to_cvxpy()

            curw = np.zeros(self.N)
            curw[-1] = 1.

            results.append(policy.values_in_time_recursive(
                t=self.returns.index[67],
                current_weights=pd.Series(
                    curw,
                    self.returns.columns),
                current_portfolio_value=1000,
                past_returns=self.returns.iloc[:67],
                past_volumes=self.volumes.iloc[:67],
                current_prices=pd.Series(1., self.volumes.columns).values)
                - curw)

        self.assertTrue(np.allclose(results[0], results[1], atol=1e-4))
        self.assertTrue(np.allclose(results[1], results[2], atol=1e-4))

        # with tcost

        results = []
        for planning_horizon in [1, 2, 5]:
            return_forecast = cvx.ReturnsForecast()
            risk_forecast = cvx.FullCovariance()
            policy = cvx.MultiPeriodOptimization(
                return_forecast
                - 10 * risk_forecast
                # - TcostModel(half_spread=5 * 1E-4)  # , power=2)
                - cvx.TransactionCost(a=25 * 1E-4, b=0.),
                constraints=[cvx.LongOnly(), cvx.LeverageLimit(1)],
                # verbose=True,
                planning_horizon=planning_horizon,
                solver=self.default_socp_solver)

            policy.initialize_estimator_recursive(
                universe=self.returns.columns,
                trading_calendar=self.returns.index)
            # policy.compile_to_cvxpy()

            curw = np.zeros(self.N)
            curw[-1] = 1.

            results.append(policy.values_in_time_recursive(
                t=self.returns.index[67],
                current_weights=pd.Series(
                    curw,
                    self.returns.columns),
                current_portfolio_value=1000,
                past_returns=self.returns.iloc[:67],
                past_volumes=self.volumes.iloc[:67],
                current_prices=pd.Series(1., self.volumes.columns)) - curw)

        self.assertFalse(np.allclose(results[0], results[1], atol=1e-4))
        self.assertFalse(np.allclose(results[1], results[2], atol=1e-4))

    def test_multi_period_optimization_syntax(self):
        """Test syntax error checks in MultiPeriodOptimization."""
        with self.assertRaises(SyntaxError):
            cvx.MultiPeriodOptimization([cvx.ReturnsForecast()], [])
        with self.assertRaises(SyntaxError):
            cvx.MultiPeriodOptimization([cvx.ReturnsForecast()], [[], []])
        with self.assertRaises(SyntaxError):
            cvx.MultiPeriodOptimization([cvx.ReturnsForecast()], None)
        with self.assertRaises(SyntaxError):
            cvx.MultiPeriodOptimization(cvx.ReturnsForecast())
        cvx.MultiPeriodOptimization(cvx.ReturnsForecast(), planning_horizon=1)

    def test_multi_period_optimization3(self):
        """Check that terminal constraint brings closer to benchmark."""

        np.random.seed(0)
        benchmark = np.random.uniform(size=self.returns.shape[1])
        benchmark /= sum(benchmark)
        benchmark = pd.Series(benchmark, self.returns.columns)

        diff_to_benchmarks = []
        for planning_horizon in [1, 2, 5]:

            return_forecast = cvx.ReturnsForecast()
            risk_forecast = cvx.FullCovariance()
            policy = cvx.MultiPeriodOptimization(
                return_forecast
                - 10 * risk_forecast
                # , power=2)
                - cvx.TransactionCost(a=5 * 1E-4, b=0.),
                constraints=[cvx.LongOnly(), cvx.LeverageLimit(1)],
                # verbose=True,
                terminal_constraint=benchmark,
                planning_horizon=planning_horizon,
                solver=self.default_socp_solver)

            policy.initialize_estimator_recursive(
                universe=self.returns.columns,
                trading_calendar=self.returns.index)
            # policy.compile_to_cvxpy()

            curw = np.zeros(self.N)
            curw[-1] = 1.

            diff_to_benchmarks.append(policy.values_in_time_recursive(
                t=self.returns.index[67],
                current_weights=pd.Series(
                    curw,
                    self.returns.columns),
                current_portfolio_value=1000,
                past_returns=self.returns.iloc[:67],
                past_volumes=self.volumes.iloc[:67],
                current_prices=pd.Series(1., self.volumes.columns))
                - benchmark)

        self.assertTrue(np.isclose(np.linalg.norm(diff_to_benchmarks[0]), 0.))
        self.assertTrue(np.linalg.norm(
            diff_to_benchmarks[0]) < np.linalg.norm(diff_to_benchmarks[1]))
        self.assertTrue(np.linalg.norm(
            diff_to_benchmarks[1]) < np.linalg.norm(diff_to_benchmarks[2]))

    def test_execute(self):
        """Test the ``execute`` method."""

        policy = cvx.Uniform()
        h = pd.Series(0., self.returns.columns)
        h.iloc[-1] = 10000
        u, t, shares_traded = policy.execute(market_data=self.market_data, h=h)
        print(t, u, shares_traded)
        self.assertTrue(np.isclose(u.sum(), 0.))
        self.assertTrue(t == self.returns.index[-1])
        self.assertTrue(len(set(u.iloc[:-1])) == 1)

        policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast() - 5 * cvx.FullCovariance(),
            [cvx.LongOnly(applies_to_cash=True)],
            solver=self.default_qp_solver)

        market_data = cvx.UserProvidedMarketData(
                    returns=self.returns, volumes=self.volumes,
                    cash_key='cash', base_location=self.datadir,
                    min_history=pd.Timedelta('0d'))

        execution = policy.execute(market_data=market_data, h=h)
        print(execution)
        u, t, shares_traded = execution
        self.assertTrue(shares_traded is None)
        self.assertTrue(np.isclose(u.sum(), 0.))
        self.assertTrue(t == self.returns.index[-1])
        self.assertTrue(u['CSCO'] >= .9)

        # wrong time
        with self.assertRaises(ValueError):
            policy.execute(
                market_data=market_data, h=h,
                t=self.returns.index[-1]+pd.Timedelta('300d'))

        h_neg_value = pd.Series(h, copy=True)
        h_neg_value.iloc[-1] = -100
        with self.assertRaises(DataError):
            execution = policy.execute(market_data=market_data, h=h_neg_value)

        h_wrong_uni = pd.Series(h[1:], copy=True)
        with self.assertRaises(ValueError):
            execution = policy.execute(market_data=market_data, h=h_wrong_uni)

        h_wrong_uni = pd.Series(h, copy=True)
        h_wrong_uni['WRONG'] = 1.
        with self.assertRaises(ValueError):
            execution = policy.execute(market_data=market_data, h=h_wrong_uni)

        h_wrong_uni = pd.Series(h, copy=True)
        h_wrong_uni.index = [el + 'SUFFIX' for el in h_wrong_uni.index]
        with self.assertRaises(ValueError):
            execution = policy.execute(market_data=market_data, h=h_wrong_uni)

        # shuffled
        execution = policy.execute(market_data=market_data, h=h)
        h_shuffled = pd.Series(h.iloc[::-1], copy=True)
        execution_shuffled = policy.execute(
            market_data=market_data, h=h_shuffled)
        self.assertTrue(np.all(execution[0] == execution_shuffled[0]))

        # for online
        market_data = cvx.UserProvidedMarketData(
                    returns=self.returns, volumes=self.volumes,
                    cash_key='cash', base_location=self.datadir,
                    min_history=pd.Timedelta('0d'),
                    online_usage=True)

        execution_online = policy.execute(market_data=market_data, h=h)
        self.assertTrue(np.all(execution[0] == execution_online[0]))


if __name__ == '__main__':

    unittest.main(warnings='error') # pragma: no cover
