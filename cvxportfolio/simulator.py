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
"""This module implements the MarketSimulator class, which strives to
simulate as accurately as possibly what would have been the realized
performance of a trading policy if it had been run in the market in the past.
In financial jargon this is called *backtesting*.
"""

import copy
import logging
import time
from pathlib import Path
import pickle
import hashlib
from functools import cached_property
from collections import defaultdict, OrderedDict
from itertools import starmap
import os

from multiprocess import Pool, Lock
import numpy as np
import pandas as pd

from .costs import BaseCost, StocksTransactionCost, StocksHoldingCost
from .data import FredTimeSeries, YfinanceTimeSeries, BASE_LOCATION
from .estimator import Estimator, DataEstimator
from .result import BacktestResult
from .utils import *
from .errors import DataError

PPY = 252
__all__ = ['StockMarketSimulator', 'MarketSimulator']


def _mp_init(l):
    global LOCK
    LOCK = l


def _hash_universe(universe):
    return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()


def _load_cache(universe, trading_frequency, base_location):
    """Load cache from disk."""
    folder = base_location / \
        f'hash(universe)={_hash_universe(universe)},trading_frequency={trading_frequency}'
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    try:
        with open(folder/'cache.pkl', 'rb') as f:
            logging.info(
                f'Loading cache for universe = {universe} and trading_frequency = {trading_frequency}')
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
    folder = base_location / \
        f'hash(universe)={_hash_universe(universe)},trading_frequency={trading_frequency}'
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


class MarketData:
    """Prepare, hold, and serve market data. 

    Not meant to be accessed by user. Most of its initialization
    is documented in MarketSimulator.    
    """

    def __init__(self,
                 universe=[],
                 returns=None,
                 volumes=None,
                 prices=None,
                 datasource='YFinance',
                 cash_key='USDOLLAR',
                 base_location=BASE_LOCATION,
                 min_history=pd.Timedelta('365.24d'),
                 # TODO change logic for this (it's now this to not drop quarterly data)
                 max_contiguous_missing='365d',
                 trading_frequency=None,
                 copy_dataframes=True,
                 **kwargs,
                 ):

        # drop duplicates and ensure ordering
        universe = sorted(set(universe))

        self.base_location = Path(base_location)
        self.min_history_timedelta = min_history
        self.max_contiguous_missing = max_contiguous_missing
        self.cash_key = cash_key

        if len(universe):
            self._get_market_data(universe, datasource)
            self._add_cash_column(self.cash_key)
            self._remove_missing_recent()
        else:
            if returns is None:
                raise SyntaxError(
                    "If you don't specify a universe you should pass `returns`.")
            self.returns = pd.DataFrame(returns, copy=copy_dataframes)
            self.volumes = volumes if volumes is None else \
                pd.DataFrame(volumes, copy=copy_dataframes)
            self.prices = prices if prices is None else \
                pd.DataFrame(prices, copy=copy_dataframes)
            if cash_key != returns.columns[-1]:
                self._add_cash_column(cash_key)

        if trading_frequency:
            self._downsample(trading_frequency)

        self._set_read_only()
        self._check_sizes()

    def _reduce_universe(self, reduced_universe):
        assert reduced_universe[-1] == self.cash_key
        logging.debug(
            f'Preparing MarketData with reduced_universe {reduced_universe}')
        return MarketData(
            returns=self.returns[reduced_universe],
            volumes=self.volumes[reduced_universe[:-1]
                                 ] if not (self.volumes is None) else None,
            prices=self.prices[reduced_universe[:-1]
                               ] if not (self.prices is None) else None,
            cash_key=self.cash_key,
            copy_dataframes=False)

    @property
    def min_history(self):
        """Min. history expressed in periods."""
        return int(np.round(self.PPY * (self.min_history_timedelta / pd.Timedelta('365.24d'))))

    @property
    def universe(self):
        return self.returns.columns

    sampling_intervals = {'weekly': 'W-MON',
                          'monthly': 'MS', 'quarterly': 'QS', 'annual': 'AS'}

    def _downsample(self, interval):
        """_downsample market data."""
        if not interval in self.sampling_intervals:
            raise SyntaxError(
                'Unsopported trading interval for down-sampling.')
        interval = self.sampling_intervals[interval]
        new_returns_index = pd.Series(self.returns.index, self.returns.index
                                      ).resample(interval, closed='left', label='left').first().values
        # print(new_returns_index)
        self.returns = np.exp(np.log(
            1+self.returns).resample(interval, closed='left', label='left'
                                     ).sum(min_count=1))-1
        self.returns.index = new_returns_index
        if self.volumes is not None:
            new_volumes_index = pd.Series(self.volumes.index, self.volumes.index
                                          ).resample(interval, closed='left', label='left').first().values
            self.volumes = self.volumes.resample(
                interval, closed='left', label='left').sum(min_count=1)
            self.volumes.index = new_volumes_index
        if self.prices is not None:
            new_prices_index = pd.Series(self.prices.index, self.prices.index
                                         ).resample(interval, closed='left', label='left').first().values
            self.prices = self.prices.resample(
                interval, closed='left', label='left').first()
            self.prices.index = new_prices_index

    @property
    def PPY(self):
        "Periods per year, assumes returns are about equally spaced."
        return periods_per_year(self.returns.index)

    def _check_sizes(self):

        if (not self.volumes is None) and (not (self.volumes.shape[1] == self.returns.shape[1] - 1)
                                           or not all(self.volumes.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Volumes should have same columns as returns, minus cash_key.')

        if (not self.prices is None) and (not (self.prices.shape[1] == self.returns.shape[1] - 1)
                                          or not all(self.prices.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Prices should have same columns as returns, minus cash_key.')

    def _serve_data_policy(self, t):
        """Give data to policy at time t."""
        tidx = self.returns.index.get_loc(t)
        past_returns = pd.DataFrame(self.returns.iloc[:tidx])
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            past_volumes = pd.DataFrame(self.volumes.iloc[:tidx])
        else:
            past_volumes = None
        current_prices = pd.Series(
            self.prices.loc[t]) if not self.prices is None else None

        return past_returns, past_volumes, current_prices

    def _serve_data_simulator(self, t):
        """Give data to simulator at time t."""
        tidx = self.returns.index.get_loc(t)
        current_and_past_returns = pd.DataFrame(self.returns.iloc[:tidx+1])
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            current_and_past_volumes = pd.DataFrame(self.volumes.iloc[:tidx+1])
        else:
            current_and_past_volumes = None
        current_prices = pd.Series(
            self.prices.loc[t]) if not self.prices is None else None

        return current_and_past_returns, current_and_past_volumes, current_prices

    def _set_read_only(self):
        """Set numpy array contained in dataframe to read only.

        This is enough to prevent direct assignement to the resulting 
        dataframe. However it could still be accidentally corrupted by assigning
        to columns or indices that are not present in the original.
        We avoid that case as well by returning a wrapped dataframe (which doesn't
        copy data on creation) in _serve_data_policy and _serve_data_simulator.
        """

        def ro(df):
            data = df.values
            data.flags.writeable = False
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.returns = ro(self.returns)

        if not self.prices is None:
            self.prices = ro(self.prices)

        if not self.volumes is None:
            self.volumes = ro(self.volumes)

    def _add_cash_column(self, cash_key):

        if not cash_key == 'USDOLLAR':
            raise NotImplementedError(
                'Currently the only data pipeline built is for USDOLLAR cash')

        data = FredTimeSeries('DFF', base_location=self.base_location)
        data._recursive_pre_evaluation()
        self.returns[cash_key] = resample_returns(
            data.data / 100, periods=self.PPY)
        self.returns[cash_key] = self.returns[cash_key].ffill()

    DATASOURCES = {'YFinance': YfinanceTimeSeries, 'FRED': FredTimeSeries}

    def _get_market_data(self, universe, datasource):
        database_accesses = {}
        print('Updating data')

        for stock in universe:
            logging.debug(
                f'Getting data for {stock} with {self.DATASOURCES[datasource]}.')
            print('.')
            database_accesses[stock] = self.DATASOURCES[datasource](
                stock, base_location=self.base_location)
            database_accesses[stock]._recursive_pre_evaluation()

        if datasource == 'YFinance':
            self.returns = pd.DataFrame(
                {stock: database_accesses[stock].data['Return'] for stock in universe})
            self.volumes = pd.DataFrame(
                {stock: database_accesses[stock].data['ValueVolume'] for stock in universe})
            self.prices = pd.DataFrame(
                {stock: database_accesses[stock].data['Open'] for stock in universe})
        else:  # only FRED for indexes
            self.prices = pd.DataFrame(
                {stock: database_accesses[stock].data for stock in universe})  # open prices
            self.returns = 1 - self.prices / self.prices.shift(-1)
            self.volumes = None

    def _remove_missing_recent(self):
        """Clean recent data.

        Yfinance has some issues with most recent data; 
        we remove recent days if there are NaNs.
        """

        if self.prices.iloc[-5:].isnull().any().any():
            logging.debug(
                'Removing some recent lines because there are missing values.')
            drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
            logging.debug(f'Dropping at index {drop_at}')
            self.returns = self.returns.loc[self.returns.index < drop_at]
            if self.prices is not None:
                self.prices = self.prices.loc[self.prices.index < drop_at]
            if self.volumes is not None:
                self.volumes = self.volumes.loc[self.volumes.index < drop_at]

        # for consistency we must also nan-out the last row of returns and volumes
        self.returns.iloc[-1] = np.nan
        if self.volumes is not None:
            self.volumes.iloc[-1] = np.nan

    def _get_backtest_times(self, start_time=None, end_time=None, include_end=True):
        """Get trading calendar from market data."""
        result = self.returns.index
        result = result[result >= self._earliest_backtest_start]
        if start_time:
            result = result[result >= start_time]
        if end_time:
            result = result[(result <= end_time)]
        if not include_end:
            result = result[:-1]
        return result

    @property
    def _break_timestamps(self):
        """List of timestamps at which a backtest should be broken.

        An asset enters into a backtest after having non-NaN returns
        for self.min_history periods and exits after having NaN returns
        for self.max_contiguous_missing. Defaults values are 252 and 10 
        respectively.
        """
        self.entry_dates = defaultdict(list)
        self.exit_dates = defaultdict(list)
        for asset in self.returns.columns[:-1]:
            single_asset_returns = self.returns[asset].dropna()
            if len(single_asset_returns) > self.min_history:
                self.entry_dates[single_asset_returns.index[self.min_history]].append(
                    asset)
                exit_date = single_asset_returns.index[-1]
                if (self.returns.index[-1] - exit_date) >= pd.Timedelta(self.max_contiguous_missing):
                    self.exit_dates[exit_date].append(asset)

        _ = sorted(set(self.exit_dates) | set(self.entry_dates))
        logging.debug(f'computing break timestamps {_}')
        return _

    @property
    def _limited_universes(self):
        """Valid universes for each section, minus cash.

        A backtest is broken into multiple ones that start at each key
        of this, have the universe specified by this, and end
        at the next startpoint.  
        """
        result = OrderedDict()
        uni = []
        for ts in self._break_timestamps:
            uni += self.entry_dates[ts]
            uni = [el for el in uni if not el in self.exit_dates[ts]]
            result[ts] = tuple(sorted(uni))
        return result

    @property
    def _earliest_backtest_start(self):
        """Earliest date at which we can start a backtest."""
        return self.returns.iloc[:, :-1].dropna(how='all').index[self.min_history]

    def _get_limited_backtests(self, start_time, end_time):
        """Get start/end times and universes of constituent backtests.

        Each one has constant universe with assets' that meet the
        ``min_history`` requirement and has not disappeared from the
        dataset.
        """

        full_backtest_times = self._get_backtest_times(start_time, end_time)
        brkt = np.array(self._break_timestamps)

        def get_valid_universe_and_its_expiration_for(time):
            try:
                return self._limited_universes[brkt[brkt <= time][-1]], \
                    brkt[brkt > time][0] if len(
                        brkt[brkt > time]) else full_backtest_times[-1]
            except IndexError:
                raise DataError(
                    'There are no assets that meet the required min_history.')

        result = []
        start = full_backtest_times[0]
        while True:

            universe, expiration = get_valid_universe_and_its_expiration_for(
                start)
            if expiration > full_backtest_times[-1]:
                expiration = full_backtest_times[-1]
            result.append({
                'start_time': start,
                'end_time': expiration,
                'universe': list(universe) + [self.cash_key]})

            if expiration == full_backtest_times[-1]:
                return result

            start = expiration


class MarketSimulator:
    """This class is a generic financial market simulator.

    It is (currently) not meant to be used directly. Look at
    :class:`StockMarketSimulator` for its version specialized
    to the stock market.
    """

    def __init__(self, universe=[], returns=None, volumes=None,
                 prices=None, costs=[], round_trades=False,
                 min_history=pd.Timedelta('365d'),
                 datasource='YFinance',
                 cash_key="USDOLLAR", base_location=BASE_LOCATION,
                 trading_frequency=None, **kwargs):
        """Initialize the Simulator and download data if necessary."""
        self.base_location = Path(base_location)

        self.enable_caching = len(universe) > 0

        self.market_data = MarketData(
            universe=universe, returns=returns,
            volumes=volumes, prices=prices,
            cash_key=cash_key, base_location=base_location,
            trading_frequency=trading_frequency,
            min_history=min_history,
            datasource=datasource,
            **kwargs)

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
        """Round dollar trade vector u.
        """
        result = pd.Series(u, copy=True)
        result.iloc[:-1] = np.round(u[:-1] / current_prices) * current_prices
        result.iloc[-1] = -sum(result.iloc[:-1])
        return result

    def _simulate(self, t, t_next, h, policy, **kwargs):
        """Get next portfolio and statistics used by Backtest for reporting.

        The signature of this method differs from other estimators
        because we pass the policy directly to it, and the past returns 
        and past volumes are computed by it.
        """

        # translate to weights
        current_portfolio_value = sum(h)
        current_weights = h / current_portfolio_value

        past_returns, past_volumes, current_prices = \
            self.market_data._serve_data_policy(t)

        # evaluate the policy
        s = time.time()
        z = policy._recursive_values_in_time(
            t=t, current_weights=current_weights,
            current_portfolio_value=current_portfolio_value,
            past_returns=past_returns, past_volumes=past_volumes,
            current_prices=current_prices, **kwargs)

        policy_time = time.time() - s

        # for safety recompute cash
        z.iloc[-1] = -sum(z.iloc[:-1])
        assert sum(z) == 0.

        # trades in dollars
        u = z * current_portfolio_value

        # get data for simulator
        current_and_past_returns, current_and_past_volumes, current_prices = \
            self.market_data._serve_data_simulator(t)

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
            periods_per_year=self.market_data.PPY,
            windowsigma=self.market_data.PPY) for cost in self.costs}

        # initialize tomorrow's holdings
        h_next = pd.Series(h_plus, copy=True)

        # debit costs from cash account
        h_next.iloc[-1] = h_plus.iloc[-1] - sum(realized_costs.values())

        # multiply positions (including cash) by market returns
        current_returns = current_and_past_returns.iloc[-1]
        h_next *= (1 + current_returns)

        return h_next, z, u, realized_costs, policy_time

    def _single_backtest(self, policy, start_time, end_time, h, universe=None):
        if universe is None:
            universe = self.market_data.universe
        backtest_times = self.market_data._get_backtest_times(
            start_time, end_time, include_end=True)

        if hasattr(policy, '_compile_to_cvxpy'):
            policy._compile_to_cvxpy()

        result = BacktestResult(universe, backtest_times, self.costs)

        # this is the main loop of a backtest
        for t, t_next in zip(backtest_times[:-1], backtest_times[1:]):
            # s = time.time()
            result.h.loc[t] = h
            h, result.z.loc[t], result.u.loc[t], realized_costs, \
                result.policy_times.loc[t] = self._simulate(
                    t=t, h=h, policy=policy, t_next=t_next)
            for cost in realized_costs:
                result.costs[cost].loc[t] = realized_costs[cost]
            result.simulator_times.loc[t] = time.time(
            ) - self.simulator_timer - result.policy_times.loc[t]
            self.simulator_timer = time.time()

        result.h.loc[pd.Timestamp(end_time)] = h

        result.cash_returns = \
            self.market_data.returns.iloc[:, -1].loc[result.u.index]

        return result

    def _concatenated_backtests(self, policy, start_time, end_time, h):
        self.simulator_timer = time.time()
        constituent_backtests_params = self.market_data._get_limited_backtests(
            start_time, end_time)
        results = []
        orig_md = self.market_data
        orig_policy = policy
        for el in constituent_backtests_params:
            logging.info(f"current universe: {el['universe']}")
            logging.info(f"interval: {el['start_time']}, {el['end_time']}")
            self.market_data = orig_md._reduce_universe(el['universe'])

            # TODO improve
            if len(el['universe']) > len(h):
                tmp = pd.Series(0., el['universe'])
                tmp.loc[h.index] = h
                h = tmp
            else:
                h = h[el['universe']]

            policy = copy.deepcopy(orig_policy)
            policy._recursive_pre_evaluation(
                universe=el['universe'],
                backtest_times=self.market_data._get_backtest_times(
                    el['start_time'], el['end_time'], include_end=True)
            )

            # if policy uses a cache load it from disk
            if hasattr(policy, 'cache') and self.enable_caching:
                logging.info('Trying to load cache from disk...')
                policy.cache = _load_cache(
                    universe=el['universe'],
                    trading_frequency=self.trading_frequency,
                    base_location=self.base_location)

            results.append(self._single_backtest(
                policy, el['start_time'], el['end_time'], h, el['universe']))

            h = results[-1].h.iloc[-1]

            # if policy used a cache write it to disk
            if hasattr(policy, 'cache') and self.enable_caching:
                logging.info('Storing cache from policy to disk...')
                _store_cache(cache=policy.cache, universe=el['universe'],
                             trading_frequency=self.trading_frequency,
                             base_location=self.base_location)

        self.market_data = orig_md

        result = self._concatenate_backtest_results(results)
        
        # temporary, will refactor these methods
        result.simulator_times.iloc[-2] += time.time() - self.simulator_timer
        return result

    def _concatenate_backtest_results(self, results):

        res = BacktestResult.__new__(BacktestResult)
        res.costs = {}

        res.h = pd.concat([el.h.iloc[:-1] if i < len(results) -
                          1 else el.h for i, el in enumerate(results)])
        for attr in ['cash_returns', 'u', 'z', 'simulator_times', 'policy_times']:
            res.__setattr__(attr, pd.concat(
                [el.__getattribute__(attr) for el in results]))

        # pandas concat can misalign the columns ordering
        ck = self.market_data.cash_key
        sortcol = sorted([el for el in res.u.columns if not el == ck]) + [ck]
        res.u = res.u[sortcol]
        res.z = res.z[sortcol]
        res.h = res.h[sortcol]
        for k in results[0].costs:
            res.costs[k] = pd.concat([el.costs[k] for el in results])

        return res

    @staticmethod
    def _worker(policy, simulator, start_time, end_time, h):
        return simulator._concatenated_backtests(policy, start_time, end_time, h)
        
        
    def optimize_hyperparameters(self, policy, start_time=None, end_time=None, 
        initial_value=1E6, h=None, objective='sharpe_ratio'):
        """Optimize hyperparameters of a policy to maximize backtest objective.
        
        EXPERIMENTAL: this method is currently being developed.
        """
        
        def modify_orig_policy(target_policy):
            hps = policy._collect_hyperparameters()
            thps = target_policy._collect_hyperparameters()
            for h,t in zip(hps, thps):
                h._index = t._index
            
        results = {}

        result_init = self.backtest(policy, start_time=start_time, end_time=end_time, 
            initial_value=initial_value, h=h)
        
        current_objective = getattr(result_init, objective)
        
        results[str(policy)] = current_objective

        for i in range(100):
            print('iteration', i)
            print('Current optimal hyper-parameters:')
            print(policy)
            print()
        
            test_policies = []
            for hp in policy._collect_hyperparameters():
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
                start_time=start_time, end_time=end_time, initial_value=initial_value, h=h)
                
            objectives_partial = [getattr(res, objective) for res in results_partial]
            
            for pol, obje in zip(test_policies, objectives_partial):
                results[str(pol)] = obje
                
            # print(results)
            
            if max(objectives_partial) <= current_objective:
                break
            
            current_objective = max(objectives_partial)   
            # policy = test_policies[np.argmax(objectives_partial)]
            modify_orig_policy(test_policies[np.argmax(objectives_partial)])
            

        
        
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

        backtest_times_inclusive = self.market_data._get_backtest_times(
            start_time, end_time, include_end=True)
        start_time = backtest_times_inclusive[0]
        end_time = backtest_times_inclusive[-1]

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

    def __init__(self, universe=[],
                 returns=None, volumes=None, prices=None,
                 costs=[StocksTransactionCost, StocksHoldingCost],
                 round_trades=True, min_history=pd.Timedelta('365d'),
                 cash_key="USDOLLAR", base_location=BASE_LOCATION,
                 trading_frequency=None, **kwargs):

        super().__init__(universe=universe,
                         returns=returns, volumes=volumes, prices=prices,
                         costs=costs, round_trades=round_trades, min_history=min_history,
                         cash_key=cash_key, base_location=base_location,
                         trading_frequency=trading_frequency, **kwargs)
