# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
# Copyright 2023- The Cvxportfolio Contributors
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


import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from .result import SimulationResult
from .costs import BaseCost
from .data import FredRate, Yfinance, TimeSeries
from .returns import MultipleReturnsForecasts, ReturnsForecast
from .estimator import Estimator, DataEstimator


class MarketSimulator(Estimator):
    """This class implements a simulator of market performance for trading strategies.

    We strive to make the parameters here as accurate as possible. The following is
    accurate as of 2023 using numbers obtained on the public website of a
    [large US-based broker](https://www.interactivebrokers.com/).


    Args:

        universe (list): list of [Yahoo Finance](https://finance.yahoo.com/) tickers on which to
            simulate performance of the trading strategy. If left unspecified you should at least
            pass `returns` and `volumes`. If you define a different market data access interface
            (look in cvxportfolio.data for how to do it) you should pass instead
            the symbol names for that data provider. Default is empty list.

        returns (pandas.DataFrame): historical open-to-open returns. Default is None, it is ignored
            if universe is specified.

        volumes (pandas.DataFrame): historical market volumes expressed in value (e.g., US dollars).
            Default is None, it is ignored if universe is specified.

        prices (pandas.DataFrame): historical open prices. Default is None, it is ignored
            if universe is specified. These are used to round the trades to integer number of stocks
            if round_trades is True, and compute per-share transaction costs (if `per_share_fixed_cost`
            is greater than zero).

        spreads (pandas.DataFrame): historical bid-ask spreads expressed as (ask-bid)/bid. Default is None,
            equivalent to 0.0. Practical spreads are negligible on US liquid stocks.

        round_trades (bool): round the trade weights provided by a policy so they correspond to an integer
            number of stocks traded. Default is True using Yahoo Finance open prices.

        per_share_fixed_cost (float): transaction cost per share traded. Default value is 0.005 (USD), uses
             Yahoo Finance open prices to simulate the number of stocks traded. See
            https://www.interactivebrokers.com/en/pricing/commissions-home.php

        transaction_cost_coefficient_b (float, pd.Series, or pd.DataFrame): coefficient that multiplies the non-linear
            term of the transaction cost. Default value is 1, you can pass any other constant value, a per-stock Series,
            or a per-day and per-stock DataFrame

        transaction_cost_exponent (float): exponent of the non-linear term of the transaction cost model. Default value 1.5,
             this is applied to the trade volume (in US dollars) over the total market volume (in US dollars). See the
            paper for more details; this model is supported by a long tradition of research in market microstructure.

        rolling_window_sigma_estimator (int): we use an historical rolling standard deviation to estimate the average
            size of the return on a stock on each day, and this multiplies the second term of the transaction cost model.
             See the paper for an explanation of the model. Here you specify the length of the rolling window to use,
             default is 1000.

        spread_on_borrowing_stocks_percent (float, pd.Series, pd.DataFrame): when shorting a stock,
            you will pay a rate on the value
            of the position equal to the cash return plus this spread, expressed in percent annualized. These
            values are hard to find historically, if you are unsure consider long-only portfolios or look
            at CFDs/futures instead. We set the default value to 0.5 (percent annualized) which is probably OK
            for US large cap stocks. See https://www.interactivebrokers.com/en/pricing/short-sale-cost.php

        spread_on_long_positions_percent (float, None, pd.Series, pd.DataFrame): if you trade CFDs you will pay interest
            on your long positions
            as well as your short positions, equal to the cash return plus this value (percent annualized). If
             instead this is None, the default value, you pay nothing on your long positions (as you do if you trade
            stocks). We don't consider dividend payments because those are already incorporated in the
            open-to-open returns as we compute them from the Yahoo Finance data. See cvxportfolio.data for details.

        dividends (float, pd.DataFrame): if not included in the returns (as they are by the default data interface,
            based on `yfinance`), you can pass a DataFrame of dividend payments which will be credited to the cash
            account (or debited, if short) at each round. Default is 0., corresponding to no dividends.

        spread_on_lending_cash_percent (float, pd.Series): the cash account will generate annualized
            return equal to the cash return minus this number, expressed in percent annualized, or zero if
            the spread is larger than the cash return. For example with USDOLLAR cash,
            if the FRED-DFF annualized rate is 4.8% and spread_on_lending_cash_percent is 0.5
            (the default value), then the uninvested cash in the portfolio generates annualized
            return of 4.3%. See https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php

        spread_on_borrowing_cash_percent (float, pd.Series): if we instead borrow cash we pay the
            cash rate plus this spread, expressed in percent annualized. Default value is 0.5.
            See https://www.interactivebrokers.com/en/trading/margin-rates.php

        cash_key (str or None): name of the cash account. Default is 'USDOLLAR', which gets downloaded by `cvxportfolio.data`
            as the Federal Funds effective rate from FRED. If None, you must pass the cash returns
            along with the stock returns as its last column.

        base_location (pathlib.Path): base location for (by default, sqlite) storage of data.
            Default is `Path.home() / "cvxportfolio"`. Unused if passing `returns` and `volumes`.
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
            rolling_window_sigma_estimator=1000,
            spread_on_borrowing_stocks_percent=.5,
            spread_on_long_positions_percent=None,
            dividends=0.,
            spread_on_lending_cash_percent=.5,
            spread_on_borrowing_cash_percent=.5,
            cash_key="USDOLLAR",
            base_location=Path.home() /
            "cvxportfolio"):
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
        self.rolling_window_sigma_estimator = rolling_window_sigma_estimator
        self.spread_on_borrowing_stocks_percent = spread_on_borrowing_stocks_percent
        self.spread_on_long_positions_percent = spread_on_long_positions_percent
        self.spread_on_lending_cash_percent = spread_on_lending_cash_percent
        self.spread_on_borrowing_cash_percent = spread_on_borrowing_cash_percent

        # compute my DataEstimator(s)
        self.sigma_estimate = DataEstimator(
            self.returns.data.iloc[:, :-1].rolling(self.rolling_window_sigma_estimator).std().shift(1))

    def prepare_data(self):
        """Build data from data storage and download interfaces.

        This is a first cut, it's doing a for loop when we could instead parallelize
        at the `yfinance` level and use the estimator logic of TimeSeries directly.
        """
        self.database_accesses = {}
        for stock in self.universe:
            print('Updating data...')
            self.database_accesses[stock] = TimeSeries(
                stock, base_location=self.base_location)
            self.database_accesses[stock].pre_evaluation()
        if not self.cash_key == 'USDOLLAR':
            raise NotImplementedError(
                'Currently the only data pipeline built is for USDOLLAR cash')
        self.database_accesses[self.cash_key] = TimeSeries(
            'DFF', source='fred', base_location=self.base_location)
        self.database_accesses[self.cash_key].pre_evaluation()

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
        """Round dollar trade vector u."""
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
            # compute number of shares traded.
            # we assume round_trades is True and we add a small number to ensure
            # we are on the safe side of rounding errors
            result += self.per_share_fixed_cost * int(sum(np.abs(u[:-1] + 1E-6) / self.prices.current_value))

        if self.spreads is not None:
            result += sum(self.spreads.current_value * np.abs(u[:-1]))/2.

        result += (np.abs(u[:-1])**self.transaction_cost_exponent) @ (self.transaction_cost_coefficient_b * self.sigma_estimate.current_value / (
            self.volumes.current_value ** (self.transaction_cost_exponent - 1)))
            
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

        # get view of past data
        past_returns = self.returns.data.loc[self.returns.data.index < t]
        past_volumes = self.volumes.data.loc[self.volumes.data.index < t]

        # update all internal estimators
        super().values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)

        # evaluate the policy
        z = policy.values_in_time(t, current_weights, current_portfolio_value, past_returns, past_volumes, **kwargs)
        # for safety
        z[-1] = -sum(z[:-1])

        # trades in dollars
        u = z * current_portfolio_value

        # zero out trades on stock that weren't trading on that day
        current_volumes = self.volumes.current_value
        non_tradable_stocks = current_volumes[current_volumes <= 0]
        u[non_tradable_stocks] = 0.

        # round trades
        if self.round_trades:
            u = self.round_trade_vector(u)

        # compute post-trade holdings (including cash balance)
        h_plus = h + u

        # we have updated the internal estimators and they are used by these
        # methods
        transaction_costs = self.transaction_costs(u)
        holding_costs = self.stocks_holding_costs(h_plus)
        cash_holding_costs = self.cash_holding_cost(h_plus)

        # multiply positions by market returns
        h_next = pd.Series(h_plus, copy=True)
        h_next[:-1] *= (1 + self.returns.current_value[:-1])
        
        # credit costs to cash (includes cash return)
        h_next[-1] = h_plus[-1] + (transaction_costs + holding_costs + cash_holding_costs)
            
        return h_next, z, u, transaction_costs, holding_costs, cash_holding_costs
        
    def initialize_policy(self, policy, start_time, end_time):
        """Initialize the policy object.
        
        This method differs from other Estimators because it is the Simulator
        that initializes the policy and all its dependents. It is called by Backtest.
        """
        policy.pre_evaluation(self.returns.data.loc[self.returns.data.index<end_time], 
                              self.volumes.data.loc[self.volumes.data.index<end_time], 
                              start_time, end_time)
        
        
    ### THE FOLLOWING METHODS ARE FROM THE ORIGINAL SIMULATOR (PRE-2023)
    ### The main difference is that you pass a list of costs, which implement
    ### a `value_expression` method, and those take care of the cost evaluation
    ### during backtest. These can be used, as they are in the examples, using
    ### `legacy_run_backtest` and `legacy_run_multiple_backtest`. 
    ###
    ### Also, we have two methods that are not part of the book
    ### and are not well-documented, `what_if` and `attribute`. We may remove
    ### all methods below this comment block in the coming months.

    def propagate(self, h, u, t):
        """Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        assert u.index.equals(h.index)

        if self.volumes is not None:
            # don't trade if volume is null
            null_trades = self.volumes.data.columns[self.volumes.data.loc[t] == 0]
            if len(null_trades):
                logging.info(
                    "No trade condition for stocks %s on %s" % (null_trades, t)
                )
                u.loc[null_trades] = 0.0

        hplus = h + u
        costs = [cost.value_expr(t, h_plus=hplus, u=u) for cost in self.costs]
        for cost in costs:
            assert not pd.isnull(cost)
            assert not np.isinf(cost)

        u[self.cash_key] = -sum(u[u.index != self.cash_key]) - sum(costs)
        hplus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        assert hplus.index.sort_values().equals(
            self.returns.data.columns.sort_values()
        )
        h_next = self.returns.data.loc[t] * hplus + hplus

        assert not h_next.isnull().values.any()
        assert not u.isnull().values.any()
        return h_next, u

    def legacy_run_backtest(
            self,
            initial_portfolio,
            start_time,
            end_time,
            policy,
            loglevel=logging.WARNING):
        """Backtest a single policy."""
        logging.basicConfig(level=loglevel)

        results = SimulationResult(
            initial_portfolio=copy.copy(initial_portfolio),
            policy=policy,
            cash_key=self.cash_key,
            simulator=self,
        )
        h = initial_portfolio

        simulation_times = self.returns.data.index[
            (self.returns.data.index >= start_time)
            & (self.returns.data.index <= end_time)
        ]
        logging.info(
            "Backtest started, from %s to %s"
            % (simulation_times[0], simulation_times[-1])
        )

        for t in simulation_times:
            logging.info("Getting trades at time %s" % t)
            start = time.time()
            try:
                u = policy.get_trades(h, t)
            except cvx.SolverError:
                logging.warning(
                    "Solver failed on timestamp %s. Default to no trades." % t
                )
                u = pd.Series(index=h.index, data=0.0)
            end = time.time()
            assert not pd.isnull(u).any()
            results.log_policy(t, end - start)

            logging.info("Propagating portfolio at time %s" % t)
            start = time.time()
            h, u = self.propagate(h, u, t)
            end = time.time()
            assert not h.isnull().values.any()
            results.log_simulation(
                t=t,
                u=u,
                h_next=h,
                risk_free_return=self.returns.data.loc[t, self.cash_key],
                exec_time=end - start,
            )

        logging.info(
            "Backtest ended, from %s to %s"
            % (simulation_times[0], simulation_times[-1])
        )
        return results

    def legacy_run_multiple_backtest(
        self,
        initial_portf,
        start_time,
        end_time,
        policies,
        loglevel=logging.WARNING,
        parallel=True,
    ):
        """Backtest multiple policies."""

        def _legacy_run_backtest(policy):
            return self.legacy_run_backtest(
                initial_portf, start_time, end_time, policy, loglevel=loglevel
            )

        num_workers = min(multiprocess.cpu_count(), len(policies))
        if parallel:
            workers = multiprocess.Pool(num_workers)
            results = workers.map(_legacy_run_backtest, policies)
            workers.close()
            return results
        else:
            return list(map(_legacy_run_backtest, policies))

    def what_if(self, time, results, alt_policies, parallel=True):
        """Run alternative policies starting from given time."""
        # TODO fix
        initial_portf = copy.copy(results.h.loc[time])
        all_times = results.h.index
        alt_results = self.legacy_run_multiple_backtest(
            initial_portf, time, all_times[-1], alt_policies, parallel
        )
        for idx, alt_result in enumerate(alt_results):
            alt_result.h.loc[time] = results.h.loc[time]
            alt_result.h.sort_index(axis=0, inplace=True)
        return alt_results

    @staticmethod
    def reduce_signal_perturb(initial_weights, delta):
        """Compute matrix of perturbed weights given initial weights."""
        perturb_weights_matrix = np.zeros(
            (len(initial_weights), len(initial_weights)))
        for i in range(len(initial_weights)):
            perturb_weights_matrix[i, :] = initial_weights / (
                1 - delta * initial_weights[i]
            )
            perturb_weights_matrix[i, i] = (1 - delta) * initial_weights[i]
        return perturb_weights_matrix

    def attribute(
            self,
            true_results,
            policy,
            selector=None,
            delta=1,
            fit="linear",
            parallel=True):
        """Attributes returns over a period to individual alpha sources.

        Args:
            true_results: observed results.
            policy: the policy that achieved the returns.
                    Alpha model must be a stream.
            selector: A map from SimulationResult to time series.
            delta: the fractional deviation.
            fit: the type of fit to perform.
        Returns:
            A dict of alpha source to return series.
        """
        # Default selector looks at profits.
        if selector is None:

            def selector(result):
                return result.v - sum(result.initial_portfolio)

        alpha_stream = policy.return_forecast
        assert isinstance(alpha_stream, MultipleReturnsForecasts)
        times = true_results.h.index
        weights = alpha_stream.weights
        assert np.sum(weights) == 1
        alpha_sources = alpha_stream.alpha_sources
        num_sources = len(alpha_sources)
        Wmat = self.reduce_signal_perturb(weights, delta)
        perturb_pols = []
        for idx in range(len(alpha_sources)):
            new_pol = copy.copy(policy)
            new_pol.return_forecast = MultipleReturnsForecasts(
                [ReturnsForecast(el.expected_returns.data) for el in alpha_sources],
                #alpha_sources,
                Wmat[idx, :]
            )
            perturb_pols.append(new_pol)
        # Simulate
        p0 = true_results.initial_portfolio
        alt_results = self.legacy_run_multiple_backtest(
            p0, times[0], times[-1], perturb_pols, parallel=parallel
        )
        # Attribute.
        true_arr = selector(true_results).values
        attr_times = selector(true_results).index
        Rmat = np.zeros((num_sources, len(attr_times)))
        for idx, result in enumerate(alt_results):
            Rmat[idx, :] = selector(result).values
        Pmat = cvx.Variable((num_sources, len(attr_times)))
        if fit == "linear":
            prob = cvx.Problem(cvx.Minimize(0), [Wmat @ Pmat == Rmat])
            prob.solve()
        elif fit == "least-squares":
            error = cvx.sum_squares(Wmat @ Pmat - Rmat)
            prob = cvx.Problem(
                cvx.Minimize(error), [
                    Pmat.T @ weights == true_arr])
            prob.solve()
        else:
            raise Exception("Unknown fitting method.")
        # Dict of results.
        wmask = np.tile(weights[:, np.newaxis], (1, len(attr_times))).T
        data = pd.DataFrame(
            columns=[s.name for s in alpha_sources],
            index=attr_times,
            data=Pmat.value.T * wmask,
        )
        data["residual"] = true_arr - \
            np.asarray((weights @ Pmat).value).ravel()
        data["RMS error"] = np.asarray(
            cvx.norm(Wmat @ Pmat - Rmat, 2, axis=0).value
        ).ravel()
        data["RMS error"] /= np.sqrt(num_sources)
        return data
