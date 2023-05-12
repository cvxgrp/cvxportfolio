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
from multiprocessing import Pool
# from multiprocess import Pool

# import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from .costs import BaseCost
from .data import FredRateTimeSeries, YfinanceTimeSeries, BASE_LOCATION
from .returns import ReturnsForecast
from .estimator import Estimator, DataEstimator
from .result import BacktestResult

__all__ = ['MarketSimulator']


def parallel_worker(policy, simulator, start_time, end_time, h):

    return simulator._single_backtest(policy, start_time, end_time, h)
    

class MarketSimulator(Estimator):
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

    periods_per_year = 252

    def __init__(
            self,
            universe=[],
            returns=None,
            volumes=None,
            costs=None,
            prices=None,
            spreads=0.,
            round_trades=True,
            per_share_fixed_cost=0.005,
            transaction_cost_coefficient_b=1.,
            transaction_cost_exponent=1.5,
            window_sigma_estimate=252,
            spread_on_borrowing_stocks_percent=.5,
            spread_on_long_positions_percent=None,
            dividends=0.,
            spread_on_lending_cash_percent=.5,
            spread_on_borrowing_cash_percent=.5,
            cash_key="USDOLLAR",
            base_location=BASE_LOCATION):
        """Initialize the Simulator and download data if necessary."""
        if not len(universe):
            if costs is None: # we allow old simulator syntax for the time being
                if ((returns is None) or (volumes is None)):
                    raise SyntaxError(
                        "If you don't specify a universe you should pass `returns` and `volumes`.")
                if not returns.shape[1] == volumes.shape[1] + 1:
                    raise SyntaxError(
                        "In `returns` you must include the cash returns as the last column (and not in `volumes`).")
            self.returns = DataEstimator(returns)
            self.volumes = DataEstimator(volumes) if not volumes is None else volumes
            self.cash_key = returns.columns[-1]
            self.costs = costs
            if not self.costs is None:
                for cost in self.costs:
                    assert isinstance(cost, BaseCost)
            self.prices = DataEstimator(prices) if prices is not None else None
            if prices is None and self.costs is None:
                if per_share_fixed_cost > 0:
                    raise SyntaxError(
                        "If you don't specify prices you can't request `per_share_fixed_cost` transaction costs.")
                if round_trades:
                    raise SyntaxError(
                        "If you don't specify prices you can't request `round_trades`.")
        else:
            self.universe = universe
            self.cash_key = cash_key
            self.base_location = base_location
            self.prepare_data()

        self.spreads = DataEstimator(spreads)
        self.dividends = DataEstimator(dividends)
        self.round_trades = round_trades
        self.per_share_fixed_cost = per_share_fixed_cost
        self.transaction_cost_coefficient_b = transaction_cost_coefficient_b
        self.transaction_cost_exponent = transaction_cost_exponent
        self.window_sigma_estimate = window_sigma_estimate
        self.spread_on_borrowing_stocks_percent = spread_on_borrowing_stocks_percent
        self.spread_on_long_positions_percent = spread_on_long_positions_percent
        self.spread_on_lending_cash_percent = spread_on_lending_cash_percent
        self.spread_on_borrowing_cash_percent = spread_on_borrowing_cash_percent

        # compute my DataEstimator(s)
        self.sigma_estimate = DataEstimator(
            self.returns.data.iloc[:, :-1].rolling(window=self.window_sigma_estimate, min_periods=1).std().shift(1))

    def prepare_data(self):
        """Build data from data storage and download interfaces.

        This is a first cut, it's doing a for loop when we could instead parallelize
        at the `yfinance` level and use the estimator logic of TimeSeries directly.
        """
        self.database_accesses = {}
        print('Updating data')
        for stock in self.universe:
            print('.')
            self.database_accesses[stock] = YfinanceTimeSeries(stock, base_location=self.base_location)
            self.database_accesses[stock].pre_evaluation()
        if not self.cash_key == 'USDOLLAR':
            raise NotImplementedError('Currently the only data pipeline built is for USDOLLAR cash')
        self.database_accesses[self.cash_key] = FredRateTimeSeries('DFF', base_location=self.base_location)
        self.database_accesses[self.cash_key].pre_evaluation()
        # print()

        # build returns
        self.returns = pd.DataFrame(
            {stock: self.database_accesses[stock].data['Return'] for stock in self.universe})
        self.returns[self.cash_key] = self.database_accesses[self.cash_key].data
        self.returns[self.cash_key] = self.returns[self.cash_key].fillna(
            method='ffill')
        

        # build volumes
        self.volumes = pd.DataFrame(
            {stock: self.database_accesses[stock].data['ValueVolume'] for stock in self.universe})
        

        # build prices
        self.prices = pd.DataFrame(
            {stock: self.database_accesses[stock].data['Open'] for stock in self.universe})
        
        
        # yfinance has some issues with most recent data; we patch it here but this
        # logic should go in .data
        if self.prices.iloc[-5:].isnull().any().any():
            drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
            self.prices = self.prices.loc[self.prices.index<drop_at]
            self.returns = self.returns.loc[self.returns.index<drop_at]
            self.volumes = self.volumes.loc[self.volumes.index<drop_at]
        
        # for consistency we must also nan-out the last row of returns and volumes
        self.returns.iloc[-1] = np.nan
        self.volumes.iloc[-1] = np.nan
        
        self.returns = DataEstimator(self.returns)
        self.volumes = DataEstimator(self.volumes)
        self.prices = DataEstimator(self.prices)
        
        
    def round_trade_vector(self, u):
        """Round dollar trade vector u.
        """
        result = pd.Series(u, copy=True)
        result[:-1] = np.round(u[:-1] / self.prices.current_value) * self.prices.current_value
        result[-1] = -sum(result[:-1])
        return result


    def transaction_costs(self, u):
        """Compute transaction costs at time t for dollar trade vector u.

        Returns a non-positive float.

        Args:
            u (pd.Series): dollar trade vector for all stocks including cash (but the cash
                term is not used here).
        """

        result = 0.
        if self.prices is not None:

            result += self.per_share_fixed_cost * int(sum(np.abs(u[:-1] + 1E-6) / self.prices.current_value))

        if self.spreads is not None:
            result += sum(self.spreads.current_value * np.abs(u[:-1]))/2.

        result += (np.abs(u[:-1])**self.transaction_cost_exponent) @ (self.transaction_cost_coefficient_b * 
            self.sigma_estimate.current_value / (
            (self.volumes.current_value+1 # we add 1 to prevent 0 volumes error
             ) ** (self.transaction_cost_exponent - 1)))
            
        assert not np.isnan(result)
        assert not np.isinf(result)
            
        return -result

    def stocks_holding_costs(self, h_plus):
        """Compute holding costs at current time for post trade holdings h_plus (only stocks).

        Args:
            h_plus (pd.Series): post trade holdings vector for all stocks including cash (but the cash
                term is not used here).
        """

        result = 0.
        cash_return = self.returns.current_value[-1]

        # shorting stocks.
        borrowed_stock_positions = np.minimum(h_plus[:-1], 0.)
        result += np.sum((cash_return +
                          (self.spread_on_borrowing_stocks_percent / 100) / self.periods_per_year) *
                         borrowed_stock_positions)

        # going long on stocks.
        if self.spread_on_long_positions_percent is not None:
            long_positions = np.maximum(h_plus[:-1], 0.)
            result -= np.sum((cash_return +
                              (self.spread_on_long_positions_percent /100) / self.periods_per_year) *
                             long_positions)

        # dividends
        result += np.sum(h_plus[:-1] * self.dividends.current_value)

        return result

    def cash_holding_cost(self, h_plus):
        """Compute holding cost on cash (including cash return) for post trade holdings h_plus."""

        cash_return = self.returns.current_value[-1]

        # we subtract from cash the value of borrowed stocks
        # if trading CFDs/futures we must amend this
        real_cash = h_plus[-1] + sum(np.minimum(h_plus[:-1], 0.))

        if real_cash > 0:
            return real_cash * \
                max(cash_return - (self.spread_on_lending_cash_percent/100)/self.periods_per_year, 0.)
        else:
            return real_cash * \
                (cash_return + (self.spread_on_borrowing_cash_percent/100)/self.periods_per_year)
                

    def simulate(self, t, h, policy, **kwargs):
        """Get next portfolio and statistics used by Backtest for reporting.

        The signature of this method differs from other estimators
        because we pass the policy directly to it, and the past returns and past volumes
        are computed by it.
        """
        
        # translate to weights
        current_portfolio_value = sum(h)
        current_weights = h / current_portfolio_value
        # print(t, current_portfolio_value)

        # get view of past data
        past_returns = self.returns.data.loc[self.returns.data.index < t]
        past_volumes = self.volumes.data.loc[self.volumes.data.index < t]

        # update internal estimators (spreads, dividends, volumes, ..., )
        super().values_in_time(t=t)

        # evaluate the policy
        z = policy.values_in_time(t=t, current_weights=current_weights, current_portfolio_value=current_portfolio_value, 
            past_returns=past_returns, past_volumes=past_volumes, current_prices=self.prices.current_value, **kwargs)
        
        # for safety recompute cash
        z[-1] = -sum(z[:-1])
        assert sum(z) == 0.
        
        # trades in dollars
        u = z * current_portfolio_value

        # zero out trades on stock that weren't trading on that day 
        current_volumes = pd.Series(self.volumes.current_value, self.volumes.data.columns)
        non_tradable_stocks = current_volumes[current_volumes <= 0].index
        u[non_tradable_stocks] = 0.

        # round trades
        if self.round_trades:
            u = self.round_trade_vector(u)
            
        # for safety recompute cash
        u[-1] = -sum(u[:-1])
        assert sum(u) == 0.

        # compute post-trade holdings (including cash balance)
        h_plus = h + u

        # we have updated the internal estimators and they are used by these methods
        transaction_costs = self.transaction_costs(u)
        holding_costs = self.stocks_holding_costs(h_plus)
        cash_holding_costs = self.cash_holding_cost(h_plus)

        # multiply positions by market returns (only non-cash)
        h_next = pd.Series(h_plus, copy=True)
        h_next[:-1] *= (1 + self.returns.current_value[:-1])
        
        # credit costs to cash (includes cash return)
        h_next[-1] = h_plus[-1] + (transaction_costs + holding_costs + cash_holding_costs)
            
        return h_next, z, u, transaction_costs, holding_costs, cash_holding_costs, 
        
    def initialize_policy(self, policy, start_time, end_time):
        """Initialize the policy object.
        """
        policy.pre_evaluation(universe = self.returns.data.columns, 
                             backtest_times = self.returns.data.index[(self.returns.data.index<end_time) & 
                                 (self.returns.data.index>=start_time)])
                                 
    def _single_backtest(self, policy, start_time, end_time, h):
        
        if hasattr(policy, 'compile_to_cvxpy'):
            policy.compile_to_cvxpy()
        
        h_df = pd.DataFrame(columns=self.returns.data.columns)
        u = pd.DataFrame(columns=self.returns.data.columns)
        z = pd.DataFrame(columns=self.returns.data.columns)
        tcost = pd.Series(dtype=float)
        hcost_stocks = pd.Series(dtype=float)
        hcost_cash = pd.Series(dtype=float)
        
        for t in self.returns.data.index[(self.returns.data.index >= start_time) & (self.returns.data.index < end_time)]:
            h_df.loc[t] = h
            h, z.loc[t], u.loc[t], tcost.loc[t], hcost_stocks.loc[t], hcost_cash.loc[t] = \
                self.simulate(t=t, h=h, policy=policy)
        
        h_df.loc[pd.Timestamp(end_time)] = h  
        
        return BacktestResult(h=h_df, u=u, z=z, tcost=tcost, hcost_stocks=hcost_stocks, hcost_cash=hcost_cash, 
            cash_returns=self.returns.data[self.cash_key].loc[u.index])
        
                  
                                 
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
            h = [h]*len(policy)
            
        if not (len(policy) == len(h)):
            raise SyntaxError("If passing lists of policies and initial portfolios they must have the same length.")
        
        # discover start and end times
        start_time = pd.Series(self.returns.data.index >= start_time, self.returns.data.index).idxmax()
        if end_time is None:
            end_time  = self.returns.data.index[-1]
        else:
            end_time = self.returns.data.index[self.returns.data.index <= end_time][-1]
        
        # initialize policies and get initial portfolios
        for i in range(len(policy)):
            self.initialize_policy(policy[i], start_time, end_time)
        
            if h[i] is None:
                h[i] = pd.Series(0., self.returns.data.columns)
                h[i][-1] = initial_value
                
        def parallel_runner(zipped):
            return self._single_backtest(zipped[0], start_time, end_time, zipped[1])
            
        # parallel_worker(policy, simulator, start_time, end_time, h)
        
        
        # decide if run in parallel or not
        if (not parallel) or len(policy) == 1:
            result = list(map(parallel_runner, zip(policy, h)))
        else:
            with Pool() as p:
                # if not __name__ == '__main__':
                #     raise SyntaxError('When executing parallel backtests, the Simulator should be instantiated ')
                result = p.starmap(parallel_worker, zip(policy, [self] * len(policy), [start_time] * len(policy), [end_time] * len(policy), h))
                
        if len(result) == 1:
            return result[0]
        return result
        

