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
"""This module implements :class:`cvxportfolio.MarketSimulator` and derived
classes. These objects model the trading activity on a financial market. They
simulate as accurately as possible what would have been the realized
performance of a trading policy if it had been run in the market in the
past. In financial jargon this is called *back-testing*.
"""

import copy
import logging
import time
from itertools import starmap
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocess import Lock, Pool

from .cache import _load_cache, _mp_init, _store_cache
from .costs import StocksHoldingCost, StocksTransactionCost
from .data import BASE_LOCATION, DownloadedMarketData, UserProvidedMarketData
from .result import BacktestResult

PPY = 252
__all__ = ['StockMarketSimulator', 'MarketSimulator']


class MarketSimulator:
    """This class is a generic financial market simulator.
    
    It can either be initialized with an instance of a class
    derived from :class:`cvxportfolio.data.MarketData`, like the two defaults
    :class:`cvxportfolio.DownloadedMarketData` and
    :class:`cvxportfolio.UserProvidedMarketData`, or with a selection of the
    arguments to the latter two.

    :param universe: List of names as understood by the data source 
        used, *e.g.*, ``['AAPL', 'GOOG']`` if using the default Yahoo Finance
        data source. If provided, a :class:`cvxportfolio.DownloadedMarketData`
        will be initialized.
    :type universe: list
    :param returns: Historical open-to-open returns. The return
        at time :math:`t` is :math:`r_t = p_{t+1}/p_t -1` where
        :math:`p_t` is the (open) price at time :math:`t`. Must
        have datetime index. If provided, a 
        :class:`cvxportfolio.UserProvidedMarketData` will be initialized. If
        universe is specified it is ignored.
    :type returns: pandas.DataFrame
    
    :param volumes: Historical market volumes, expressed in units
        of value (*e.g.*, US dollars). If
        universe is specified it is ignored.
    :type volumes: pandas.DataFrame or None
    :param prices: Historical open prices (*e.g.*, used for rounding
        trades in the :class:`cvxportfolio.MarketSimulator`). If
        universe is specified it is ignored.
    :type prices: pandas.DataFrame or None
    
    :param datasource: The data source used, if providing a universe
        (for :class:`cvxportfolio.DownloadedMarketData`).
    :type datasource: str or :class:`cvxportfolio.data.SymbolData` class
        (not instance)
    
    :param cash_key: Name of the cash account. Its returns
        are the risk-free rate.
    :type cash_key: str

    :param trading_frequency: Instead of using frequency implied by
        the index of the returns, down-sample all dataframes.
        We implement ``'weekly'``, ``'monthly'``, ``'quarterly'`` and
        ``'annual'``. By default (None) don't down-sample. 
    :type trading_frequency: str or None
    
    :param market_data: An instance of a :class:`cvxportfolio.data.MarketData`
        derived class. If provided, all previous arguments are ignored.
    :type market_data: :class:`cvxportfolio.data.MarketData` instance or None
    :param base_location: The location of the storage. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_location: pathlib.Path
    
    :param costs: List of costs that are applied at each time in a back-test.
    :type costs: list of :class:`cvxportfolio.costs.SimulatorCost` classes or
        instances
    
    :param round_trades: If market prices are available, round trades to
        integer number of shares.
    :type round_trades: bool
    
 
    """

    def __init__(self, universe=(), returns=None, volumes=None,
                 prices=None, market_data=None, costs=(), round_trades=False,
                 datasource='YahooFinance',
                 cash_key="USDOLLAR",
                 base_location=BASE_LOCATION,
                 trading_frequency=None):
        """Initialize the Simulator and download data if necessary."""
        self.base_location = Path(base_location)

        if not market_data is None:
            self.market_data = market_data
        else:

            if not len(universe) and prices is None:
                if round_trades:
                    raise SyntaxError(
                        "If you don't specify prices you can't request "
                        + "`round_trades`.")

            if not len(universe):
                self.market_data = UserProvidedMarketData(
                    returns=returns,
                    volumes=volumes, prices=prices,
                    cash_key=cash_key,
                    base_location=base_location,
                    trading_frequency=trading_frequency)
            else:
                self.market_data = DownloadedMarketData(
                    universe=universe,
                    cash_key=cash_key,
                    base_location=base_location,
                    trading_frequency=trading_frequency,
                    datasource=datasource)

        self.round_trades = round_trades
        self.costs = [el() if isinstance(el, type) else el for el in costs]
        # self.lock = Lock()
       # self.kwargs = kwargs

    @staticmethod
    def _round_trade_vector(u, current_prices):
        """Round dollar trade vector u."""
        result = pd.Series(u, copy=True)
        result.iloc[:-1] = np.round(u[:-1] / current_prices) * current_prices
        result.iloc[-1] = -sum(result.iloc[:-1])
        return result

    def simulate(self, t, t_next, h, policy, past_returns, current_returns,
                past_volumes, current_volumes, current_prices):
        """Get next portfolio and statistics used by Backtest for reporting.

        The signature of this method differs from other estimators
        because we pass the policy directly to it, and the past returns
        and past volumes are computed by it.
        """

        # translate to weights
        current_portfolio_value = sum(h)
        current_weights = pd.to_numeric(h / current_portfolio_value)

        # evaluate the policy
        s = time.time()
        policy_w = policy.values_in_time_recursive(
            t=t, current_weights=current_weights,
            current_portfolio_value=current_portfolio_value,
            past_returns=past_returns, past_volumes=past_volumes,
            current_prices=current_prices)

        z = policy_w - current_weights

        policy_time = time.time() - s

        # for safety recompute cash
        z.iloc[-1] = -sum(z.iloc[:-1])
        assert sum(z) == 0.

        # trades in dollars
        u = z * current_portfolio_value

        # zero out trades on stock that weren't trading on that day
        if not (current_volumes is None):
            non_tradable_stocks = current_volumes[current_volumes <= 0].index
            if len(non_tradable_stocks):
                logging.info(
                    f"At time {t} the simulator canceled trades on assets {non_tradable_stocks}"
                    " because their market volumes for the period are zero.")
            u[non_tradable_stocks] = 0.

        # round trades
        if self.round_trades:
            u = self._round_trade_vector(u, current_prices)

        # recompute cash
        u.iloc[-1] = -sum(u.iloc[:-1])
        assert sum(u) == 0.

        # compute post-trade holdings (including cash balance)
        h_plus = h + u

        # evaluate cost functions
        realized_costs = {cost.__class__.__name__: cost.simulate(
            t=t, u=u,  h_plus=h_plus,
            past_volumes=past_volumes,
            current_volumes=current_volumes,
            past_returns=past_returns,
            current_returns=current_returns,
            current_prices=current_prices,
            t_next=t_next,
            periods_per_year=self.market_data.periods_per_year,
            windowsigma=self.market_data.periods_per_year)
                for cost in self.costs}

        # initialize tomorrow's holdings
        h_next = pd.Series(h_plus, copy=True)

        # debit costs from cash account
        h_next.iloc[-1] = h_plus.iloc[-1] - sum(realized_costs.values())

        # multiply positions (including cash) by market returns
        assert not np.any(current_returns.isnull())
        h_next *= (1 + current_returns)

        return h_next, z, u, realized_costs, policy_time

    def _get_initialized_policy(self, orig_policy, universe, trading_calendar):

        policy = copy.deepcopy(orig_policy)

        policy.initialize_estimator_recursive(
            universe=universe, trading_calendar=trading_calendar)

        # if policy uses a cache load it from disk
        if hasattr(policy, '_cache'):
            logging.info('Trying to load cache from disk...')
            policy._cache = _load_cache(
              signature=self.market_data.partial_universe_signature(universe),
              base_location=self.base_location)

        # if hasattr(policy, 'compile_to_cvxpy'):
        #     policy.compile_to_cvxpy()

        return policy

    def _finalize_policy(self, policy, universe):
        if hasattr(policy, '_cache'):
            logging.info('Storing cache from policy to disk...')
            _store_cache(
              cache=policy._cache,
              signature=self.market_data.partial_universe_signature(universe),
              base_location=self.base_location)

    def _backtest(self, policy, start_time, end_time, h):
        """Run a backtest with changing universe."""

        timer = time.time()

        trading_calendar = self.market_data.trading_calendar(
            start_time, end_time, include_end=True)

        _, current_returns, _, _, _ = self.market_data.serve(
            trading_calendar[0])
        universe = current_returns.index

        used_policy = self._get_initialized_policy(
            policy, universe=universe, trading_calendar=trading_calendar)

        result = BacktestResult(
            universe=universe, trading_calendar=trading_calendar,
            costs=self.costs)

        for t, t_next in zip(trading_calendar[:-1], trading_calendar[1:]):

            past_returns, current_returns, past_volumes, current_volumes, \
                 current_prices = self.market_data.serve(t)
            current_universe = current_returns.index

            if not current_universe.equals(h.index):

                self._finalize_policy(used_policy, h.index)

                h = self._adjust_h_new_universe(h, current_universe)
                used_policy = self._get_initialized_policy(
                    policy, universe=current_universe,
                    trading_calendar=trading_calendar[trading_calendar >= t])

            h_next, z, u, realized_costs, policy_time = self.simulate(
                t=t, h=h, policy=used_policy,
                t_next=t_next,
                past_returns=past_returns,
                current_returns=current_returns,
                past_volumes=past_volumes,
                current_volumes=current_volumes,
                current_prices=current_prices)

            if hasattr(used_policy, 'benchmark'):
                w_bm = used_policy.benchmark.current_value
                bm_ret = w_bm @ current_returns
            else:
                bm_ret = None

            simulator_time = time.time() - timer - policy_time

            timer = time.time()

            result._log_trading(t=t, h=h, z=z, u=u, costs=realized_costs,
                                policy_time=policy_time,
                                simulator_time=simulator_time,
                                cash_return=current_returns.iloc[-1],
                                benchmark_return=bm_ret)

            h = h_next

            if sum(h) <= 0.: # bankruptcy
                logging.warning(f'Back-test ended in bankruptcy at time {t}!')
                break

        self._finalize_policy(used_policy, h.index)

        result._log_final(t, t_next, h,
            extra_simulator_time=time.time() - timer)

        return result

    # def _single_backtest(self, policy, start_time, end_time, h, universe=None):
    #     if universe is None:
    #         universe = self.market_data.universe
    #     trading_calendar = self.market_data._get_trading_calendar(
    #         start_time, end_time, include_end=True)
    #
    #     if hasattr(policy, 'compile_to_cvxpy'):
    #         policy.compile_to_cvxpy()
    #
    #     result = BacktestResult(universe, trading_calendar, self.costs)
    #
    #     # this is the main loop of a backtest
    #     for t, t_next in zip(trading_calendar[:-1], trading_calendar[1:]):
    #         # s = time.time()
    #         result.h.loc[t] = h
    #         h, result.z.loc[t], result.u.loc[t], realized_costs, \
    #             result.policy_times.loc[t] = self.simulate(
    #                 t=t, h=h, policy=policy, t_next=t_next)
    #         for cost in realized_costs:
    #             result.costs[cost].loc[t] = realized_costs[cost]
    #         result.simulator_times.loc[t] = time.time(
    #         ) - self.simulator_timer - result.policy_times.loc[t]
    #         self.simulator_timer = time.time()
    #
    #     result.h.loc[pd.Timestamp(end_time)] = h
    #
    #     result.cash_returns =\
    #         self.market_data.returns.iloc[:, -1].loc[result.u.index]
    #
    #     return result

    def _adjust_h_new_universe(self, h, new_universe):
        """Adjust holdings vector for change in universe.
        
        :param h: (Pre-trade) holdings vector in value units (e.g., USDOLLAR).
            Its index is the trading universe in the period before the present one.
        :type h: pd.Series
        :param new_universe: New trading universe for the current trading period.
        :type new_universe: pd.Index
        
        :returns: new pre-trade holdings vector with index is ``new_universe``
        :rtype: pd.Series
        
        For any new asset that is present in ``new_universe`` but not in ``h.index``
        we set the corrensponding value of ``h`` to 0. Any removed asset that is present in 
        ``h.index`` instead is removed from h and its value is added to the cash account.
        
        Note that we ignore the transaction cost involved in liquidating the position. 
        You can redefine this method in a derived class to change this behavior.
        """

        # check that cash key didn't change
        assert new_universe[-1] == h.index[-1]

        intersection = pd.Index(set(new_universe).intersection(h.index))
        new_h = pd.Series(0., new_universe)
        new_h[intersection] = h[intersection]

        new_assets = pd.Index(set(new_universe).difference(h.index))
        if len(new_assets):
            logging.info(f'Adjusting h vector by adding assets {new_assets}')

        remove_assets = pd.Index(set(h.index).difference(new_universe))
        if len(remove_assets):
            total_liquidation = h[remove_assets].sum()
            logging.info(
                f"Adjusting h vector by removing assets {remove_assets}."
                " Their current market value of {total_liquidation} is added"
                " to the cash account.")
            new_h.iloc[-1] += total_liquidation

        return new_h

    @staticmethod
    def _worker(policy, simulator, start_time, end_time, h):
        return simulator._backtest(policy, start_time, end_time, h)

    def optimize_hyperparameters(self, policy, start_time=None, end_time=None,
        initial_value=1E6, h=None, objective='sharpe_ratio', parallel=True):
        """Optimize hyperparameters of a policy to maximize back-test objective.

        :param policy: Trading policy with symbolic hyperparameters.
        :type policy: cvx.BaseTradingPolicy
        :param start_time: start time of the back-test on which we optimize; 
            if market it close, the first trading day after it is selected.
        :type start_time: str or datetime
        :param end_time: End time of the back-test on which we optimize;  
            if market it closed, the last trading day before it is selected
        :type end_time: str or datetime or None
        :param initial_value: Initial value in dollar of the portfolio, if not specifying
            ``h`` it is assumed the initial portfolio is all cash; if ``h`` is specified
            this is ignored.
        :type initial_value: float
        :param h: Initial portfolio ``h`` expressed in dollar positions. If ``None``
            an initial portfolio of ``initial_value`` in cash is used.
        :type h: pd.Series or None
        
        :param objective: Attribute of :class:`BacktestResult` that is maximized.
        :type objective: str

        :returns: The provided policy, whose hyperparameters have optimal values.
        :rtype: cvxportfolio.Policy
        """

        def modify_orig_policy(target_policy):
            hps = policy.collect_hyperparameters()
            thps = target_policy.collect_hyperparameters()
            for h, t in zip(hps, thps):
                h._index = t._index

        results = {}

        current_result = self.backtest(policy, start_time=start_time,
            end_time=end_time,
            initial_value=initial_value, h=h)

        current_objective = getattr(current_result, objective)

        results[str(policy)] = current_objective

        for i in range(100):
            print('iteration', i)
            # print('Current optimal hyper-parameters:')
            # print(policy)
            print('Current objective:')
            print(current_objective)
            # print()
            # print('Current result:')
            # print(current_result)
            # print()

            test_policies = []
            for hp in policy.collect_hyperparameters():
                try:
                    hp._increment()
                    if not (str(policy) in results):
                        test_policies.append(copy.deepcopy(policy))
                    hp._decrement()
                except IndexError:
                    pass
                try:
                    hp._decrement()
                    if not (str(policy) in results):
                        test_policies.append(copy.deepcopy(policy))
                    hp._increment()
                except IndexError:
                    pass

            if not len(test_policies):
                break

            results_partial = self.backtest_many(test_policies,
                start_time=start_time, end_time=end_time,
                initial_value=initial_value,
                h=h, parallel=parallel)

            objectives_partial = [getattr(res, objective)
                for res in results_partial]

            for pol, obje in zip(test_policies, objectives_partial):
                results[str(pol)] = obje

            # print(results)

            if max(objectives_partial) <= current_objective:
                break

            current_objective = max(objectives_partial)
            # policy = test_policies[np.argmax(objectives_partial)]
            modify_orig_policy(test_policies[np.argmax(objectives_partial)])
            current_result = results_partial[np.argmax(objectives_partial)]

    def backtest(self, policy, start_time=None, end_time=None, initial_value=1E6, h=None):
        """Backtest trading policy.

        The default initial portfolio is all cash, or you can pass any portfolio with
        the `h` argument.

        :param policy: trading policy
        :type policy: cvx.BaseTradingPolicy
        :param start_time: start time of the backtest; if market it close, the first trading day
             after it is selected
        :type start_time: str or datetime
        :param end_time: end time of the backtest; if market it close, the last trading day
             before it is selected
        :type end_time: str or datetime or None
        :param initial_value: initial value in dollar of the portfolio, if not specifying
            ``h`` it is assumed the initial portfolio is all cash; if ``h`` is specified
            this is ignored
        :type initial_value: float
        :param h: initial portfolio ``h`` expressed in dollar positions. If ``None``
            an initial portfolio of ``initial_value`` in cash is used.
        :type h: pd.Series or None

        :returns result: instance of :class:`BacktestResult` which has all relevant backtest
            data and logic to compute metrics, generate plots, ...
        :rtype result: cvx.BacktestResult
        """
        return self.backtest_many([policy], start_time=start_time, end_time=end_time,
                                  initial_value=initial_value, h=None if h is None else [h], parallel=False)[0]

    def backtest_many(self, policies, start_time=None, end_time=None, initial_value=1E6, h=None, parallel=True):
        """Backtest many trading policies.

        The default initial portfolio is all cash, or you can pass any portfolio with
        the `h` argument, or a list of those.

        :param policies: trading policies
        :type policy: list of cvx.BaseTradingPolicy:
        :param start_time: start time of the backtests; if market it close, the first trading day
            after it is selected. Currently it is not possible to specify different start times
            for different policies, so the same is used for all.
        :type start_time: str or datetime
        :param end_time: end time of the backtests; if market it close, the last trading day
            before it is selected. Currently it is not possible to specify different end times
            for different policies, so the same is used for all.
        :type end_time: str or datetime or None
        :param initial_value: initial value in dollar of the portfolio, if not specifying
            ``h`` it is assumed the initial portfolio is all cash; if ``h`` is specified
            this is ignored
        :param h: initial portfolio `h` expressed in dollar positions, or list of those. If
            passing a list it must have the same lenght as the policies. If this argument
            is specified, ``initial_value`` is ignored, otherwise the same portfolio of
            ``initial_value`` all in cash is used as starting point for all backtests.
        :type h: list or pd.Series or None
        :param parallel: whether to run in parallel. If runnning in parallel you **must be careful
            at how you use this method**. If you use this in a script, you *should* define the MarketSimulator
            *in* the `if __name__ == '__main__:'` clause, and call this method there as well.
        :type parallel: bool

        :returns result: list of instances of :class:`BacktestResult` which have all relevant backtest
            data and logic to compute metrics, generate plots, ...
        :rtype result: list of cvx.BacktestResult
        """

        if not hasattr(policies, '__len__'):
            raise SyntaxError('You should pass a list of policies.')

        if not hasattr(h, '__len__'):
            h = [h] * len(policies)

        if not (len(policies) == len(h)):
            raise SyntaxError(
                'If passing lists of policies and initial portfolios they must have the same length.')

        if start_time is not None:
            start_time = pd.Timestamp(start_time)
            if start_time.tz is None:
                start_time = start_time.tz_localize(
                    self.market_data.trading_calendar().tz)

        if end_time is not None:
            end_time = pd.Timestamp(end_time)
            if end_time.tz is None:
                end_time = end_time.tz_localize(
                    self.market_data.trading_calendar().tz)

        trading_calendar_inclusive = self.market_data.trading_calendar(
            start_time, end_time, include_end=True)
        start_time = trading_calendar_inclusive[0]
        end_time = trading_calendar_inclusive[-1]
        _, initial_returns, _, _, _ = self.market_data.serve(start_time)
        initial_universe = initial_returns.index

        # initialize policies and get initial portfolios
        for i in range(len(policies)):
            if h[i] is None:
                h[i] = pd.Series(0., initial_universe)
                h[i].iloc[-1] = initial_value

        n = len(policies)

        zip_args = zip(policies, [self] * n,
                       [start_time] * n, [end_time] * n, h)

        if (not parallel) or len(policies) == 1:
            result = list(starmap(self._worker, zip_args))
        else:
            with Pool(initializer=_mp_init, initargs=(Lock(),)) as p:
                result = p.starmap(self._worker, zip_args)

        return [el for el in result]


class StockMarketSimulator(MarketSimulator):
    """This class implements a simulator of the stock market.
    It simplifies the interface of :class:`MarketSimulator` and
    has (realistic) default costs.

    :param universe: List of Yahoo Finance stock names.
    :type universe: list
    
    :param cash_key: Name of the cash account. Its returns
        are the risk-free rate.
    :type cash_key: str

    :param trading_frequency: Instead of using frequency implied by
        the index of the returns, down-sample all dataframes.
        We implement ``'weekly'``, ``'monthly'``, ``'quarterly'`` and
        ``'annual'``. By default (None) don't down-sample. 
    :type trading_frequency: str or None
    
    :param base_location: The location of the storage. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_location: pathlib.Path
    
    :param costs: List of costs that are applied at each time in a back-test.
        By default we add :class:`StocksTransactionCost` 
        and :class:`StocksHoldingCost`.
    :type costs: list of :class:`SimulatorCost` classes or instances
    
    :param round_trades: Round trades to integer number of shares. By default, True.
    :type round_trades: bool
    
    :param kwargs: You can add any other argument to pass to 
        :class:`MarketSimulator`'s initializer.
    :type kwargs: dict
    """

    def __init__(self, universe=(),
                 costs=(StocksTransactionCost, StocksHoldingCost),
                 round_trades=True,
                 cash_key="USDOLLAR", base_location=BASE_LOCATION,
                 trading_frequency=None, **kwargs):

        super().__init__(universe=universe,
                         costs=costs, round_trades=round_trades,
                         cash_key=cash_key, base_location=base_location,
                         trading_frequency=trading_frequency, **kwargs)
