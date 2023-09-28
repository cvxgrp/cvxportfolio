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
"""This module implements the MarketSimulator class, which strives to.

simulate as accurately as possibly what would have been the realized
performance of a trading policy if it had been run in the market in the
past. In financial jargon this is called *backtesting*.
"""

import copy
import hashlib
import logging
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from functools import cached_property
from itertools import starmap
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocess import Lock, Pool

from .costs import BaseCost, StocksHoldingCost, StocksTransactionCost
from .data import (BASE_LOCATION, DownloadedMarketData, FredSymbolData,
                   UserProvidedMarketData, YahooFinanceSymbolData)
from .errors import DataError
from .estimator import DataEstimator, Estimator
from .result import BacktestResult
from .utils import (periods_per_year_from_datetime_index, repr_numpy_pandas,
                    resample_returns)

PPY = 252
__all__ = ['StockMarketSimulator', 'MarketSimulator']


def _mp_init(l):
    global LOCK
    LOCK = l


def _hash_universe(universe):
    return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()


def _load_cache(universe, trading_frequency, base_location):
    """Load cache from disk."""
    folder = base_location / (
        f'hash(universe)={_hash_universe(universe)},'
        + f'trading_frequency={trading_frequency}')
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    try:
        with open(folder/'cache.pkl', 'rb') as f:
            logging.info(
                f'Loading cache for universe = {universe}'
                f' and trading_frequency = {trading_frequency}')
            return pickle.load(f)
    except FileNotFoundError:
        logging.info(f'Cache not found!')
        return {}
    finally:
        if 'LOCK' in globals():
            logging.debug(f'Releasing cache lock from process {os.getpid()}')
            LOCK.release()


def _store_cache(cache, universe, trading_frequency, base_location):
    """Store cache to disk."""
    folder = base_location / (
        f'hash(universe)={_hash_universe(universe)},'
        f'trading_frequency={trading_frequency}')
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    folder.mkdir(exist_ok=True)
    with open(folder/'cache.pkl', 'wb') as f:
        logging.info(
            f'Storing cache for universe = {universe} and trading_frequency = {trading_frequency}')
        pickle.dump(cache, f)
    if 'LOCK' in globals():
        logging.debug(f'Releasing cache lock from process {os.getpid()}')
        LOCK.release()


class MarketSimulator:
    """This class is a generic financial market simulator.

    It is (currently) not meant to be used directly. Look at
    :class:`StockMarketSimulator` for its version specialized
    to the stock market.
    """

    def __init__(self, universe=(), returns=None, volumes=None,
                 prices=None, market_data=None, costs=(), round_trades=False,
                 min_history=pd.Timedelta('365d'),
                 datasource='YFinance',
                 cash_key="USDOLLAR", base_location=BASE_LOCATION,
                 trading_frequency=None,
                 copy_dataframes=True):
        """Initialize the Simulator and download data if necessary."""
        self.base_location = Path(base_location)

        self.enable_caching = len(universe) > 0

        if not market_data is None:
            self.market_data = market_data
        else:
            if not len(universe):
                self.market_data = UserProvidedMarketData(
                    returns=returns,
                    volumes=volumes, prices=prices,
                    cash_key=cash_key,
                    base_location=base_location,
                    trading_frequency=trading_frequency,
                    min_history=min_history,
                    copy_dataframes=copy_dataframes)
            else:
                self.market_data = DownloadedMarketData(
                    universe=universe,
                    cash_key=cash_key,
                    base_location=base_location,
                    trading_frequency=trading_frequency,
                    min_history=min_history,
                    datasource=datasource)

        self.trading_frequency = trading_frequency

        if not len(universe) and prices is None:
            if round_trades:
                raise SyntaxError(
                    "If you don't specify prices you can't request "
                    + "`round_trades`.")

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

    def _simulate(self, t, t_next, h, policy, mask=None, **kwargs):
        """Get next portfolio and statistics used by Backtest for reporting.

        The signature of this method differs from other estimators
        because we pass the policy directly to it, and the past returns
        and past volumes are computed by it.
        """

        # translate to weights
        current_portfolio_value = sum(h)
        current_weights = h / current_portfolio_value

        past_returns, past_volumes, current_prices =\
            self.market_data._serve_data_policy(t, mask=mask)

        # evaluate the policy
        s = time.time()
        policy_w = policy.values_in_time_recursive(
            t=t, current_weights=current_weights,
            current_portfolio_value=current_portfolio_value,
            past_returns=past_returns, past_volumes=past_volumes,
            current_prices=current_prices, **kwargs)

        z = policy_w - current_weights

        policy_time = time.time() - s

        # for safety recompute cash
        z.iloc[-1] = -sum(z.iloc[:-1])
        assert sum(z) == 0.

        # trades in dollars
        u = z * current_portfolio_value

        # get data for simulator
        current_and_past_returns, current_and_past_volumes, current_prices =\
            self.market_data._serve_data_simulator(t, mask=mask)

        # zero out trades on stock that weren't trading on that day
        if not (current_and_past_volumes is None):
            current_volumes = current_and_past_volumes.iloc[-1]
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
        realized_costs = {cost.__class__.__name__: cost._simulate(
            t=t, u=u,  h_plus=h_plus,
            current_and_past_volumes=current_and_past_volumes,
            current_and_past_returns=current_and_past_returns,
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
        current_returns = current_and_past_returns.iloc[-1]
        assert not np.any(current_returns.isnull())
        h_next *= (1 + current_returns)

        return h_next, z, u, realized_costs, policy_time

    def _get_initialized_policy(self, orig_policy, universe, trading_calendar):

        policy = copy.deepcopy(orig_policy)

        policy.initialize_estimator_recursive(
            universe=universe, trading_calendar=trading_calendar)

        # if policy uses a cache load it from disk
        if hasattr(policy, 'cache') and self.enable_caching:
            logging.info('Trying to load cache from disk...')
            policy.cache = _load_cache(
                universe=universe,
                trading_frequency=self.trading_frequency,
                base_location=self.base_location)

        if hasattr(policy, 'compile_to_cvxpy'):
            policy.compile_to_cvxpy()

        return policy

    def _finalize_policy(self, policy, universe):
        if hasattr(policy, 'cache') and self.enable_caching:
            logging.info('Storing cache from policy to disk...')
            _store_cache(cache=policy.cache, universe=universe,
                          trading_frequency=self.trading_frequency,
                          base_location=self.base_location)

    def _concatenated_backtests(self, policy, start_time, end_time, h):
        """Run a backtest with changing universe."""

        timer = time.time()

        trading_calendar = self.market_data.trading_calendar(
            start_time, end_time, include_end=True)

        universe = self.market_data._universe_at_time(trading_calendar[0])

        used_policy = self._get_initialized_policy(
            policy, universe=universe, trading_calendar=trading_calendar)

        result = BacktestResult(
            universe=universe, trading_calendar=trading_calendar, costs=self.costs)

        for t, t_next in zip(trading_calendar[:-1], trading_calendar[1:]):

            current_universe = self.market_data._universe_at_time(t)

            if not current_universe.equals(h.index):

                self._finalize_policy(used_policy, h.index)

                h = self._adjust_h_new_universe(h, current_universe)
                used_policy = self._get_initialized_policy(
                    policy, universe=current_universe,
                    trading_calendar=trading_calendar[trading_calendar >= t])

            h_next, z, u, realized_costs, policy_time = self._simulate(
                t=t, h=h, policy=used_policy,
                t_next=t_next, mask=current_universe)

            simulator_time = time.time() - timer - policy_time

            timer = time.time()

            result._log_trading(t=t, h=h, z=z, u=u, costs=realized_costs,
                                policy_time=policy_time,
                                simulator_time=simulator_time)

            h = h_next

        self._finalize_policy(used_policy, h.index)

        result.cash_returns = self.market_data.returns.iloc[:, -1].loc[result.u.index]

        result.h.loc[pd.Timestamp(trading_calendar[-1])] = h

        result.simulator_times.loc[pd.Timestamp(trading_calendar[-2])] += time.time() - timer

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
    #             result.policy_times.loc[t] = self._simulate(
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

    # def _concatenate_backtest_results(self, results):
    #
    #     res = BacktestResult.__new__(BacktestResult)
    #     res.costs = {}
    #
    #     res.h = pd.concat([el.h.iloc[:-1] if i < len(results) -
    #                       1 else el.h for i, el in enumerate(results)])
    #     for attr in ['cash_returns', 'u', 'z', 'simulator_times', 'policy_times']:
    #         res.__setattr__(attr, pd.concat(
    #             [el.__getattribute__(attr) for el in results]))
    #
    #     # pandas concat can misalign the columns ordering
    #     ck = self.market_data.cash_key
    #     sortcol = sorted([el for el in res.u.columns if not el == ck]) + [ck]
    #     res.u = res.u[sortcol]
    #     res.z = res.z[sortcol]
    #     res.h = res.h[sortcol]
    #     for k in results[0].costs:
    #         res.costs[k] = pd.concat([el.costs[k] for el in results])
    #
    #     return res

    def _adjust_h_new_universe(self, h: pd.Series, new_universe: pd.Index) -> pd.Series:
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
            logging.info(f"Adjusting h vector by removing assets {remove_assets}."
                " Their current market value of {total_liquidation} is added"
                " to the cash account.")
            new_h.iloc[-1] += total_liquidation

        return new_h

    @staticmethod
    def _worker(policy, simulator, start_time, end_time, h):
        return simulator._concatenated_backtests(policy, start_time, end_time, h)

    def optimize_hyperparameters(self, policy, start_time=None, end_time=None,
        initial_value=1E6, h=None, objective='sharpe_ratio', parallel=True):
        """Optimize hyperparameters of a policy to maximize backtest objective.

        EXPERIMENTAL: this method is currently being developed.
        """

        def modify_orig_policy(target_policy):
            hps = policy.collect_hyperparameters()
            thps = target_policy.collect_hyperparameters()
            for h, t in zip(hps, thps):
                h._index = t._index

        results = {}

        current_result = self.backtest(policy, start_time=start_time, end_time=end_time,
            initial_value=initial_value, h=h)

        current_objective = getattr(current_result, objective)

        results[str(policy)] = current_objective

        for i in range(100):
            print('iteration', i)
            print('Current optimal hyper-parameters:')
            print(policy)
            print('Current objective:')
            print(current_objective)
            print()
            print('Current result:')
            print(current_result)
            print()

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
                start_time=start_time, end_time=end_time, initial_value=initial_value,
                h=h, parallel=parallel)

            objectives_partial = [getattr(res, objective) for res in results_partial]

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

        # TODO: here put get_trading_calendar
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

        # initialize policies and get initial portfolios
        for i in range(len(policies)):
            if h[i] is None:
                h[i] = pd.Series(0., self.market_data.universe)
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

    We strive to make the parameters here as accurate as possible. The following is
    accurate as of 2023 using numbers obtained on the public website of a
    `large US-based broker <https://www.interactivebrokers.com/>`_.

    :param universe: list of `Yahoo Finance <https://finance.yahoo.com/>`_ tickers on which to
        simulate performance of the trading strategy. If left unspecified you should at least
        pass `returns` and `volumes`. If you define a different market data access interface
        (look in `cvxportfolio.data` for how to do it) you should pass instead
        the symbol names for that data provider. Default is empty list.
    :type universe: list or None
    :param returns: historical open-to-open returns. Default is None, it is ignored
        if universe is specified.
    :type returns: pandas.DataFrame
    :param volumes: historical market volumes expressed in value (e.g., US dollars).
            Default is None, it is ignored if universe is specified.
    :type volumes: pandas.DataFrame
    :param prices: historical open prices. Default is None, it is ignored
        if universe is specified. These are used to round the trades to integer number of stocks
        if round_trades is True, and compute per-share transaction costs (if `per_share_fixed_cost`
        is greater than zero).
    :type prices: pandas.DataFrame
    :param round_trades: round the trade weights provided by a policy so they correspond to an integer
        number of stocks traded. Default is True using Yahoo Finance open prices.
    :type round_trades: bool
    :param min_history: minimum history required for a stock to be included in a backtest. The stock
        will be ignored for this amount of time after its IPO, and then be included.
    :type min_history: pandas.Timedelta
    :param costs: list of BaseCost instances or class objects. If class objects (the default) they will
        be instantiated internally with their default arguments.
    :type costs: list
    :param cash_key: name of the cash account. Default is 'USDOLLAR', which gets downloaded by `cvxportfolio.data`
        as the Federal Funds effective rate from FRED. If None, you must pass the cash returns
        along with the stock returns as its last column.
    :type cash_key: str or None
    :param base_location: base location for storage of data.
        Default is `Path.home() / "cvxportfolio_data"`. Unused if passing `returns` and `volumes`.
    :type base_location: pathlib.Path or str:
    :param trading_frequency: optionally choose a different frequency for
        trades than the one of the data used.
        The default interface (Yahoo finance) provides daily trading data,
        and so that is the default frequency for trades. With this argument you can set instead
        the trading frequency to ``"weekly"``, which trades every Monday (or the first
        non-holiday trading day of each week), ``"monthly"``, which trades every first of the month (ditto),
        ``"quarterly"``, and ``"annual"``.
    :type trading_frequency: str or None
    """

    def __init__(self, universe=(),
                 returns=None, volumes=None, prices=None,
                 costs=(StocksTransactionCost, StocksHoldingCost),
                 round_trades=True, min_history=pd.Timedelta('365d'),
                 cash_key="USDOLLAR", base_location=BASE_LOCATION,
                 trading_frequency=None, **kwargs):

        super().__init__(universe=universe,
                         returns=returns, volumes=volumes, prices=prices,
                         costs=costs, round_trades=round_trades, min_history=min_history,
                         cash_key=cash_key, base_location=base_location,
                         trading_frequency=trading_frequency, **kwargs)
