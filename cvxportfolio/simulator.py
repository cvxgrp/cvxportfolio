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
from multiprocess import Pool
import pickle
import hashlib
from functools import cached_property
from collections import defaultdict

import numpy as np
import pandas as pd

from .costs import BaseCost
from .data import FredRateTimeSeries, YfinanceTimeSeries, BASE_LOCATION
from .estimator import Estimator, DataEstimator
from .result import BacktestResult

PPY = 252
__all__ = ['MarketSimulator', 'simulate_cash_holding_cost', 'simulate_stocks_holding_cost', 'simulate_transaction_cost', 'MarketData']


def parallel_worker(policy, simulator, start_time, end_time, h):
    return simulator._single_backtest(policy, start_time, end_time, h)
    

def hash_universe(universe):
    return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()
        
    
def load_cache(universe, base_location):
    folder = base_location/f'hash(universe)={hash_universe(universe)}'
    try:
        with open(folder/'cache.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
    
def store_cache(cache, universe, base_location):
    folder = base_location/f'hash(universe)={hash_universe(universe)}'
    folder.mkdir(exist_ok=True)
    with open(folder/'cache.pkl', 'wb') as f:
        pickle.dump(cache, f)
    
    
def simulate_cash_holding_cost(t, h_plus, current_and_past_returns,
                                spread_on_lending_cash_percent=0.5, 
                                spread_on_borrowing_cash_percent=0.5, 
                                periods_per_year=PPY, **kwargs):
    """Simulate holding cost for the cash account. 
    
    TODO move arguments documentation from MarketSimulator.
    """
                                
    multiplier = 1 / (100 * periods_per_year)
    lending_spread = DataEstimator(spread_on_lending_cash_percent).values_in_time(t) * multiplier
    borrowing_spread = DataEstimator(spread_on_borrowing_cash_percent).values_in_time(t) * multiplier

    cash_return = current_and_past_returns.iloc[-1,-1]
    real_cash = h_plus.iloc[-1] + sum(np.minimum(h_plus.iloc[:-1], 0.))

    if real_cash > 0:
        return real_cash * (max(cash_return - lending_spread, 0.) - cash_return)
    else:
        return real_cash * borrowing_spread


def simulate_stocks_holding_cost(t, h_plus, current_and_past_returns,
                                spread_on_borrowing_stocks_percent=0.5, 
                                dividends=0., periods_per_year=PPY, **kwargs):
    """Holding cost for stocks. 
    
    TODO move arguments documentation from MarketSimulator.
    """
                                
    multiplier = 1 / (100 * periods_per_year)
    borrowing_spread = DataEstimator(spread_on_borrowing_stocks_percent).values_in_time(t) * multiplier
    dividends = DataEstimator(dividends).values_in_time(t)
    result = 0.
    cash_return = current_and_past_returns.iloc[-1,-1]
    borrowed_stock_positions = np.minimum(h_plus.iloc[:-1], 0.)
    result += np.sum((cash_return + borrowing_spread) * borrowed_stock_positions)
    result += np.sum(h_plus[:-1] * DataEstimator(dividends).values_in_time(t))
    return result
              
            
def simulate_transaction_cost(t, u, current_prices, current_and_past_volumes,
                            current_and_past_returns, persharecost=0.005,
                            linearcost=0., nonlinearcoefficient=1.,
                            windowsigma=PPY, exponent=1.5, **kwargs):    
    """Transaction cost model for the market simulator.
    
    TODO move arguments documentation from MarketSimulator
    """
        
    persharecost = DataEstimator(persharecost).values_in_time(t) if not \
        (persharecost is None) else None
    nonlinearcoefficient = DataEstimator(nonlinearcoefficient).values_in_time(t) if not \
        (nonlinearcoefficient is None) else None
        
    sigma = np.std(current_and_past_returns.iloc[-windowsigma:, :-1], axis=0)

    result = 0.
    if not (persharecost is None):
        if current_prices is None:
            raise SyntaxError("If you don't provide prices you should set persharecost to None")
        result += persharecost * int(sum(np.abs(u.iloc[:-1] + 1E-6) / current_prices.values))

    result += sum(DataEstimator(linearcost).values_in_time(t) * np.abs(u.iloc[:-1]))

    if not (nonlinearcoefficient is None):
        if current_and_past_volumes is None:
            raise SyntaxError("If you don't provide volumes you should set nonlinearcoefficient to None")
        # we add 1 to the volumes to prevent 0 volumes error (trades are cancelled on 0 volumes)
        result += (np.abs(u.iloc[:-1])**exponent) @ (nonlinearcoefficient  *
            sigma / ((current_and_past_volumes.iloc[-1] + 1) ** (exponent - 1)))

    assert not np.isnan(result)
    assert not np.isinf(result)

    return -result
        
        
class MarketData:
    """Prepare, hold, and serve market data.
    
    Arguments are detailed in MarketSimulator, which initializes this.
    """
    
    def __init__(self, 
        universe = [], 
        returns=None,
        volumes=None,
        prices=None, 
        cash_key='USDOLLAR',
        base_location=BASE_LOCATION, 
        min_history=PPY,
        max_contiguous_missing='10d',
        **kwargs,
    ):
        
        # drop duplicates and ensure ordering
        universe = sorted(set(universe))
        
        self.base_location = Path(base_location)
        self.min_history = min_history
        self.max_contiguous_missing = max_contiguous_missing
        
        if len(universe):
            self.get_market_data(universe)
            self.add_cash_column(cash_key)
        else:
            if returns is None:
                raise SyntaxError("If you don't specify a universe you should pass `returns`.")
            # if not returns.shape[1] == volumes.shape[1] + 1:
            #     raise SyntaxError(
            #         "In `returns` you must include the cash returns as the last column (and not in `volumes`).")
            self.returns = returns
            self.volumes = volumes
            self.prices = prices
            if cash_key != returns.columns[-1]:
                self.add_cash_column(cash_key)
        
        self.set_read_only()
        self.check_sizes()
            
    @property
    def universe(self):
        return self.returns.columns
        
    @cached_property
    def PPY(self):
        "Periods per year, assumes returns are about equally spaced."
        idx = self.returns.index
        return int(np.round(len(idx) / ((idx[-1] - idx[0]) / pd.Timedelta('365.24d'))))
        
    
    def check_sizes(self):
        
        if (not self.volumes is None) and (not (self.volumes.shape[1] == self.returns.shape[1] - 1) \
            or not all(self.volumes.columns == self.returns.columns[:-1])):
            raise SyntaxError('Volumes should have same columns as returns, minus cash_key.')
        
        if (not self.prices is None) and (not (self.prices.shape[1] == self.returns.shape[1] - 1) \
            or not all(self.prices.columns == self.returns.columns[:-1])):
            raise SyntaxError('Prices should have same columns as returns, minus cash_key.')            
        
        
    def serve_data_policy(self, t):
        """Give data to policy at time t."""
        
        tidx = self.returns.index.get_loc(t)
        past_returns = pd.DataFrame(self.returns.iloc[:tidx])
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            past_volumes = pd.DataFrame(self.volumes.iloc[:tidx]) 
        else:
            past_volumes = None
        current_prices = pd.Series(self.prices.loc[t]) if not self.prices is None else None
        
        return past_returns, past_volumes, current_prices
        
    
    def serve_data_simulator(self, t):
        """Give data to simulator at time t."""
        
        tidx = self.returns.index.get_loc(t)
        current_and_past_returns = pd.DataFrame(self.returns.iloc[:tidx+1])
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            current_and_past_volumes = pd.DataFrame(self.volumes.iloc[:tidx+1])
        else:
            current_and_past_volumes = None
        current_prices = pd.Series(self.prices.loc[t]) if not self.prices is None else None
        
        return current_and_past_returns, current_and_past_volumes, current_prices
        
        
    def set_read_only(self):
        """Set numpy array contained in dataframe to read only.
        
        This is enough to prevent direct assignement to the resulting 
        dataframe. However it could still be accidentally corrupted by assigning
        to columns or indices that are not present in the original.
        We avoid that case as well by returning a wrapped dataframe (which doesn't
        copy data on creation) in serve_data_policy and serve_data_simulator.
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
            
    def add_cash_column(self, cash_key):
        
        if not cash_key == 'USDOLLAR':
            raise NotImplementedError('Currently the only data pipeline built is for USDOLLAR cash')
            
        # TODO do rate transformation here with periods_per_year
        data = FredRateTimeSeries('DFF', base_location=self.base_location)
        data.pre_evaluation()
        self.returns[cash_key] = data.data
        self.returns[cash_key] = self.returns[cash_key].fillna(method='ffill')
        
    
    def get_market_data(self, universe):
        
        database_accesses = {}
        print('Updating data')
        for stock in universe:
            print('.')
            database_accesses[stock] = YfinanceTimeSeries(stock, base_location=self.base_location)
            database_accesses[stock].pre_evaluation()

        self.returns = pd.DataFrame({stock: database_accesses[stock].data['Return'] for stock in universe})
        self.volumes = pd.DataFrame({stock: database_accesses[stock].data['ValueVolume'] for stock in universe})
        self.prices = pd.DataFrame({stock: database_accesses[stock].data['Open'] for stock in universe})
        
        self.remove_missing_recent()
                
        
    def remove_missing_recent(self):
            
        # yfinance has some issues with most recent data; 
        # we remove recent days if there are NaNs
        if self.prices.iloc[-5:].isnull().any().any():
            drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
            self.prices = self.prices.loc[self.prices.index<drop_at]
            self.returns = self.returns.loc[self.returns.index<drop_at]
            self.volumes = self.volumes.loc[self.volumes.index<drop_at]
        
        # for consistency we must also nan-out the last row of returns and volumes
        self.returns.iloc[-1] = np.nan
        self.volumes.iloc[-1] = np.nan
        
        
    def backtest_times(self, start_time=None, end_time=None, include_end=False):
        """Get trading calendar from market data."""
        result = self.returns.index
        if start_time:
            result = result[result >= start_time]
        if end_time:
            result = result[(result <= end_time)]
        if not include_end:
            result = result[:-1]
        return result

    @cached_property
    def break_timestamps(self):
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
                self.entry_dates[single_asset_returns.index[self.min_history]].append(asset)
                exit_date = single_asset_returns.index[-1]
                if (self.returns.index[-1] - exit_date) >= pd.Timedelta(self.max_contiguous_missing):
                    self.exit_dates[exit_date].append(asset) 

        return sorted(set(self.exit_dates) | set(self.entry_dates))
        
    @cached_property    
    def limited_universes(self):
        """Valid universes for each section, minus cash."""
        result = {}
        uni = []
        for ts in self.break_timestamps:
            uni += self.entry_dates[ts]
            uni = [el for el in uni if not el in self.exit_dates[ts]]
            result[ts] = tuple(sorted(uni))
        return result
        
        
    # def break_up_backtest(self, start_time=None, end_time=None, min_history=252):
    #     """Break up backtest into smaller backtests with constant universe.
    #     """
    #
    #     backtest_times = self.backtest_times(start_time, end_time)
    #     breakpoints = [el for el in self.break_timestamps if
    #         (el > backtest_times[0]) and el <= backtest_times[-1]]
    #
    #     if not len(breakpoints):
    #         return backtest_times
    #
    #     results = [backtest_times[backtest_times < breakpoints[0]]]
    #
    #     for i in range(len(breakpoints)-1):
    #         results.append(
    #         backtest_times[(backtest_times >= breakpoints[i]) & (backtest_times < breakpoints[i+1])]
    #         )
    #
    #     results += [backtest_times[backtest_times >= breakpoints[-1]]]
    #
    #     return results
        
    

        

class MarketSimulator:
    """This class implements a simulator of market performance for trading strategies.

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
    :param spreads: historical bid-ask spreads expressed as (ask-bid)/bid. Default is None,
        equivalent to 0.0. Practical spreads are negligible on US liquid stocks.
    :type spreads: pandas.DataFrame
    :param round_trades: round the trade weights provided by a policy so they correspond to an integer
        number of stocks traded. Default is True using Yahoo Finance open prices.
    :type round_trades: bool
    :param per_share_fixed_cost: transaction cost per share traded. Default value is 0.005 (USD), uses
        Yahoo Finance open prices to simulate the number of stocks traded. See 
        `this page <https://www.interactivebrokers.com/en/pricing/commissions-home.php>`_.
    :type per_share_fixed_cost: float
    :param transaction_cost_coefficient_b: coefficient that multiplies the non-linear
        term of the transaction cost. Default value is 1, you can pass any other constant value, a per-stock Series,
        or a per-day and per-stock DataFrame
    :type transaction_cost_coefficient_b: float, pd.Series, or pd.DataFrame
    :param transaction_cost_exponent: exponent of the non-linear term of the transaction cost model. Default value 1.5,
        this is applied to the trade volume (in US dollars) over the total market volume (in US dollars). See the
        paper for more details; this model is supported by a long tradition of research in market microstructure.
    :type transaction_cost_exponent: float
    :param window_sigma_estimate: we use an historical rolling standard deviation to estimate the average
        size of the return on a stock on each day, and this multiplies the second term of the transaction cost model.
        See the paper for an explanation of the model. Here you specify the length of the rolling window to use,
        default is 252 (typical number of trading days in a year).
    :type window_sigma_estimate: int
    :param spread_on_borrowing_stocks_percent: when shorting a stock,
        one pays a rate on the value of the position equal to the cash return plus this spread, 
        expressed in percent annualized. These values are hard to find historically, if you are unsure consider 
        long-only portfolios or look at CFDs/futures instead. We set the default value to 0.5 (percent annualized) 
        which is probably OK for US large cap stocks. See `this page <https://www.interactivebrokers.com/en/pricing/short-sale-cost.php>`_.
    :type spread_on_borrowing_stocks_percent: float, pd.Series, pd.DataFrame
    :param spread_on_long_positions_percent: if you trade CFDs one pays interest on long positions  as well as your short positions, 
        equal to the cash return plus this value (percent annualized). If
        instead this is None, the default value, you pay nothing on your long positions (as you do if you trade
        stocks). We don't consider dividend payments because those are already incorporated in the
        open-to-open returns as we compute them from the Yahoo Finance data. See cvxportfolio.data for details.
    :type spread_on_long_positions_percent: float, None, pd.Series, pd.DataFrame
    :param dividends: if not included in the returns (as they are by the default data interface,
        based on `yfinance`), you can pass a DataFrame of dividend payments which will be credited to the cash
        account (or debited, if short) at each round. Default is 0., corresponding to no dividends.
    :type dividends: float, pd.DataFrame
    :param spread_on_lending_cash_percent: the cash account will generate annualized
        return equal to the cash return minus this number, expressed in percent annualized, or zero if
        the spread is larger than the cash return. For example with USDOLLAR cash,
        if the FRED-DFF annualized rate is 4.8% and spread_on_lending_cash_percent is 0.5
        (the default value), then the uninvested cash in the portfolio generates annualized
        return of 4.3%. See `this page <https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php>_`.
    :type spread_on_lending_cash_percent: float, pd.Series
    :param spread_on_borrowing_cash_percent: if one instead borrows cash he pays the
        cash rate plus this spread, expressed in percent annualized. Default value is 0.5.
        See `this page <https://www.interactivebrokers.com/en/trading/margin-rates.php>_`.
    :type spread_on_borrowing_cash_percent: float, pd.Series 
    :param cash_key: name of the cash account. Default is 'USDOLLAR', which gets downloaded by `cvxportfolio.data`
        as the Federal Funds effective rate from FRED. If None, you must pass the cash returns
        along with the stock returns as its last column.
    :type cash_key: str or None
    :param base_location: base location for storage of data.
        Default is `Path.home() / "cvxportfolio_data"`. Unused if passing `returns` and `volumes`.
    :type base_location: pathlib.Path or str: 
    """

    def __init__(
        self,
        universe=[],
        returns=None,
        volumes=None,
        prices=None,
        # costs=[simulate_transaction_cost, simulate_stocks_holding_cost],
        round_trades=True,
        # spread_on_lending_cash_percent=.5,
        # spread_on_borrowing_cash_percent=.5,
        min_history_for_inclusion=PPY,
        cash_key="USDOLLAR",
        base_location=BASE_LOCATION,
        # periods_per_year=PPY,
        **kwargs,
    ):
        """Initialize the Simulator and download data if necessary."""
        self.base_location = Path(base_location)
        # self.periods_per_year = periods_per_year
        
        self.market_data = MarketData(
            universe=universe, returns=returns,
            volumes=volumes, prices=prices,
            cash_key=cash_key, base_location=base_location,
            # periods_per_year=self.periods_per_year, 
            **kwargs)
                
        # if not len(universe):
        #     if ((returns is None) or (volumes is None)):
        #         raise SyntaxError(
        #             "If you don't specify a universe you should pass `returns` and `volumes`.")
        #     if not returns.shape[1] == volumes.shape[1] + 1:
        #         raise SyntaxError(
        #            "In `returns` you must include the cash returns as the last column (and not in `volumes`).")
        #    self.returns = DataEstimator(returns)
        #    self.volumes = DataEstimator(volumes) if not volumes is None else volumes
        #    self.cash_key = returns.columns[-1]
        #    self.prices = DataEstimator(prices) if prices is not None else None
        if not len(universe) and prices is None:
            if round_trades:
                raise SyntaxError(
                    "If you don't specify prices you can't request `round_trades`.")
       #  else:
            # self.universe = universe
            # self.cash_key = cash_key
            # self.prepare_data()

        self.round_trades = round_trades
        self.min_history_for_inclusion = min_history_for_inclusion
        
        # self.costs = costs
        self.costs = [simulate_transaction_cost, simulate_stocks_holding_cost, simulate_cash_holding_cost]
        self.kwargs = kwargs

        
    @staticmethod
    def round_trade_vector(u, current_prices):
        """Round dollar trade vector u.
        """
        result = pd.Series(u, copy=True)
        result[:-1] = np.round(u[:-1] / current_prices) * current_prices
        result[-1] = -sum(result[:-1])
        return result


    def simulate(self, t, h, policy, **kwargs):
        """Get next portfolio and statistics used by Backtest for reporting.

        The signature of this method differs from other estimators
        because we pass the policy directly to it, and the past returns and past volumes
        are computed by it.
        """
        
        # translate to weights
        current_portfolio_value = sum(h)
        current_weights = h / current_portfolio_value

        past_returns, past_volumes, current_prices = self.market_data.serve_data_policy(t)

        # update internal estimators (spreads, dividends, volumes, ..., )
        # TODO this should be superflous now
        # super().values_in_time(t=t)

        # evaluate the policy
        s = time.time()
        z = policy.values_in_time(t=t, current_weights=current_weights, current_portfolio_value=current_portfolio_value, 
            past_returns=past_returns, past_volumes=past_volumes, current_prices=current_prices, **kwargs)
        policy_time = time.time() - s
        
        # for safety recompute cash
        z[-1] = -sum(z[:-1])
        assert sum(z) == 0.
        
        # trades in dollars
        u = z * current_portfolio_value
        
        # get data for simulator
        current_and_past_returns, current_and_past_volumes, current_prices = self.market_data.serve_data_simulator(t)

        # zero out trades on stock that weren't trading on that day 
        current_volumes = current_and_past_volumes.iloc[-1]
        non_tradable_stocks = current_volumes[current_volumes <= 0].index
        u[non_tradable_stocks] = 0.

        # round trades
        if self.round_trades:
            u = self.round_trade_vector(u, current_prices)
            
        # for safety recompute cash
        u[-1] = -sum(u[:-1])
        assert sum(u) == 0.

        # compute post-trade holdings (including cash balance)
        h_plus = h + u

        # we have updated the internal estimators and they are used by these methods
        transaction_costs = self.costs[0](t, u, 
            current_prices=current_prices, current_and_past_volumes=current_and_past_volumes, 
            windowsigma=self.market_data.PPY,
            current_and_past_returns=current_and_past_returns, **self.kwargs)
        holding_costs = self.costs[1](
            t=t, h_plus=h_plus, current_and_past_returns=current_and_past_returns,
            periods_per_year=self.market_data.PPY, **self.kwargs)
        cash_holding_costs = self.costs[2](
            t=t, h_plus=h_plus, current_and_past_returns=current_and_past_returns,
            periods_per_year=self.market_data.PPY, **self.kwargs)
        
        # initialize tomorrow's holdings
        h_next = pd.Series(h_plus, copy=True)
        
        # credit costs to cash
        h_next[-1] = h_plus[-1] + (transaction_costs + holding_costs + cash_holding_costs)

        # multiply positions by market returns
        current_returns = current_and_past_returns.iloc[-1]
        h_next *= (1 + current_returns)
            
        return h_next, z, u, transaction_costs, holding_costs, cash_holding_costs, policy_time
        
    def initialize_policy(self, policy, start_time, end_time):
        """Initialize the policy object.
        """
        policy.pre_evaluation(universe = self.market_data.universe,
                             backtest_times = self.market_data.backtest_times(start_time, end_time))
        
        # if policy initialized a cache, rewrite it with loaded one
        if hasattr(policy, 'cache'):
            policy.cache = load_cache(universe=self.market_data.universe, base_location=self.base_location)
            
                                 
    def _single_backtest(self, policy, start_time, end_time, h):
        
        # self.initialize_policy(policy, start_time, end_time)
        
        if hasattr(policy, 'compile_to_cvxpy'):
            policy.compile_to_cvxpy()
        
        h_df = pd.DataFrame(columns=self.market_data.universe)
        u = pd.DataFrame(columns=self.market_data.universe)
        z = pd.DataFrame(columns=self.market_data.universe)
        tcost = pd.Series(dtype=float)
        hcost_stocks = pd.Series(dtype=float)
        hcost_cash = pd.Series(dtype=float)
        policy_times = pd.Series(dtype=float)
        simulator_times = pd.Series(dtype=float)
        
        for t in self.market_data.backtest_times(start_time, end_time):
            h_df.loc[t] = h
            s = time.time()
            h, z.loc[t], u.loc[t], tcost.loc[t], hcost_stocks.loc[t], hcost_cash.loc[t], policy_times.loc[t] = \
                self.simulate(t=t, h=h, policy=policy)
            simulator_times.loc[t] = time.time() - s - policy_times.loc[t]
        
        h_df.loc[pd.Timestamp(end_time)] = h  
        
        if hasattr(policy, 'cache'):
            store_cache(cache=policy.cache, universe=self.market_data.universe, base_location=self.base_location)
        
        return BacktestResult(h=h_df, u=u, z=z, tcost=tcost, hcost_stocks=hcost_stocks, hcost_cash=hcost_cash, 
            cash_returns=self.market_data.returns.iloc[:,-1].loc[u.index], 
                policy_times=policy_times, simulator_times=simulator_times)
        
                  
                                 
    def backtest(self, policy, start_time, end_time=None, initial_value = 1E6, h=None, parallel=True):
        """Backtest one or more trading policy.
        
        If runnning in parallel you must be careful at how you use this method. If 
        you use this in a script, you should define the MarketSimulator
        *in* the `if __name__ == '__main__:'` clause, and call this method there as well.
        
        The default initial portfolio is all cash, or you can pass any portfolio with
        the `h` argument, or a list of those if running multiple backtests.
        
        :param policy: if passing a single trading policy it performs a single backtest 
            in the main process. If passing a list, it uses Python multiprocessing to
            create multiple processes and run many policies in parallel.
        :type policy: cvx.BaseTradingPolicy or list
        :param start_time: start time of the backtest(s), if holiday, the first trading day
             after it is selected
        :type start_time: str or datetime 
        :param end_time: end time of the backtest(s), if holiday, the last trading day
             before it is selected
        :type end_time: str or datetime or None
        :param initial_value: initial value in dollar of the portfolio, if not specifying
            `h` it is assumed the initial portfolio is all cash
        :type initial_value: float
        :param h: initial portfolio `h` expressed in dollar positions, or list of those 
            for multiple backtests
        :type h: list or pd.Series or None
        :param parallel: whether to run in parallel if there are multiple policies or not.
            If not, it just iterates through the policies in the main process.
        :type parallel: bool
        
        :returns result: instance of :class:`BacktestResult` which has all relevant backtest
            data and logic to compute metrics, generate plots, ...
        :rtype result: cvx.BacktestResult or list
        """
        
        # turn policy and h into lists
        if not hasattr(policy, '__len__'):
            policy = [policy]
        
        if not hasattr(h, '__len__'):
            h = [h] * len(policy)
            
        if not (len(policy) == len(h)):
            raise SyntaxError("If passing lists of policies and initial portfolios they must have the same length.")
        
        # discover start and end times
        # start_time = pd.Series(self.returns.data.index >= start_time, self.returns.data.index).idxmax()
        # if end_time is None:
        #     end_time  = self.returns.data.index[-1]
        # else:
        #    end_time = self.returns.data.index[self.returns.data.index <= end_time][-1]
            
        backtest_times_inclusive = self.market_data.backtest_times(start_time, end_time, include_end=True)
        start_time = backtest_times_inclusive[0]
        end_time = backtest_times_inclusive[-1]
            
        # TODO fix this - discard names that don't meet the min_history_for_inclusion
        history = (~self.market_data.returns.loc[self.market_data.returns.index < start_time].isnull()).sum()
        reduced_universe = self.market_data.returns.columns[history >= self.min_history_for_inclusion]
        self.market_data.returns = self.market_data.returns[reduced_universe]
        self.market_data.volumes = self.market_data.volumes[reduced_universe[:-1]]
        self.market_data.prices = self.market_data.prices[reduced_universe[:-1]]
        # self.sigma_estimate.data = self.sigma_estimate.data[reduced_universe[:-1]]
        
        # initialize policies and get initial portfolios
        for i in range(len(policy)):
            self.initialize_policy(policy[i], start_time, end_time)
        
            if h[i] is None:
                h[i] = pd.Series(0., self.market_data.universe)
                h[i][-1] = initial_value
                
        def nonparallel_runner(zipped):
            return self._single_backtest(zipped[0], start_time, end_time, zipped[1])
                    
        # decide if run in parallel or not
        if (not parallel) or len(policy)==1: 
            result = list(map(nonparallel_runner, zip(policy, h)))
        else:
            with Pool() as p:
                result = p.starmap(parallel_worker, zip(policy, [self] * len(policy), [start_time] * len(policy), [end_time] * len(policy), h))
           
                
        if len(result) == 1:
            return result[0]
        return result
        

