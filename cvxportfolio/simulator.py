# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
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

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from .returns import MultipleReturnsForecasts

from .result import SimulationResult
from .costs import BaseCost
from .data import FredRate, Yfinance

# TODO update benchmark weights (?)
# Also could try jitting with numba.


class BaseMarketSimulator:
    """Base class for market simulators.

    Each derived class implements specific usecases, such as stock portfolio simualtors,
    CFDs portfolios, or futures portfolios. One might also subclass this (or one of the derived classes)
    to specialize the simulator to a specific regional market, or a specific asset class.
    """


class NewMarketSimulator(BaseMarketSimulator):
    """This class implements a simulator of market performance for trading strategies.

    We strive to make the parameters here as accurate as possible. The following is
    accurate as of 2023Q2 using numbers obtained on the public website of a
    [large US-based broker](https://www.interactivebrokers.com/).

    Attributes:
        cash_keys (dict): registers a cash_key name with a data reader and a symbol name.
            By default we provide the USDOLLAR cash account whose rate is the effective
            fund rate by the US-fed (fred). If you use MarketSimulator to simulate
            performance of portfolios where the cash account is not in USD, say in EUR
            or something else, you'd have to build a datareader like we did for FRED
            and provide the right symbol to look up.

    Args:

        universe (list): list of [Yahoo Finance](https://finance.yahoo.com/) tickers on which to
            simulate performance of the trading strategy. If left unspecified you should at least
            pass `returns`. If you define a different market data access interface
            (look in cvxportfolio.data for how to do it) you should pass instead
            the symbol names for that data provider. Default is empty list.

        returns (pandas.DataFrame): historical open-to-open returns. Default is None, it is ignored
            if universe is specified.

        volumes (pandas.DataFrame): historical market volumes expressed in value (e.g., US dollars).
            Default is None, it is ignored if universe is specified.

        prices (pandas.DataFrame): historical open prices. Default is None, it is ignored
            if universe is specified. These are used to round the trades to integer number of stocks
            if round_trades is True, and compute per-share transaction costs.

        spreads (pandas.DataFrame): historical bid-ask spreads expressed as (ask-bid)/bid. Default is zero,
            practical spreads are negligible on US liquid stocks.

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

        spread_on_borrowing_stocks_percent (float): when shorting a stock, you will pay a rate on the value
            of the position equal to the cash return plus this spread, expressed in percent annualized. These
            values are hard to find historically, if you are unsure consider long-only portfolios or look
            at CFDs/futures instead. We set the default value to 0.5 (percent annualized) which is probably
            OK for US large caps. See https://www.interactivebrokers.com/en/pricing/short-sale-cost.php

        spread_on_long_positions_percent (float or None): if you trade CFDs you will pay interest on your long positions
            as well as your short positions, equal to the cash return plus this value (percent annualized). If
             instead this is None, the default value, you pay nothing on your long positions (as you do if you trade
            stocks). We don't consider dividend payments because those are already incorporated in the
            open-to-open returns as we compute them from the Yahoo Finance data. See cvxportfolio.data for details.

        spread_on_lending_cash_percent (float): the cash account will generate annualized
            return equal to the cash return minus this number, expressed in percent annualized, or zero if
            the spread is larger than the cash return. For example with USDOLLAR cash,
            if the FRED-DFF annualized rate is 4.8% and spread_on_lending_cash_percent is 0.5
            (the default value), then the uninvested cash in the portfolio generates annualized
            return of 4.3%. See https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php

        spread_on_borrowing_cash_percent (float): if we instead borrow cash we pay the
            cash rate plus this spread, expressed in percent annualized. Default value is 0.5.
            See https://www.interactivebrokers.com/en/trading/margin-rates.php

        cash_key (str): name of the cash account, there must be a matching data reader and symbol in
            MarketSimulator.cash_keys. Default is 'USDOLLAR'.
    """

    cash_keys = {"USDOLLAR": (FredRate, "DFF")}


class MarketSimulator(BaseMarketSimulator):
    """Current market simulator, name will change soon."""

    logger = None

    def __init__(
        self,
        market_returns,
        costs,
        market_volumes=None,
        cash_key="cash",
    ):
        """Provide market returns object and cost objects."""
        self.market_returns = market_returns
        if market_volumes is not None:
            self.market_volumes = market_volumes[
                market_volumes.columns.difference([cash_key])
            ]
        else:
            self.market_volumes = None

        self.costs = costs
        for cost in self.costs:
            assert isinstance(cost, BaseCost)

        self.cash_key = cash_key

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

        if self.market_volumes is not None:
            # don't trade if volume is null
            null_trades = self.market_volumes.columns[self.market_volumes.loc[t] == 0]
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
            self.market_returns.columns.sort_values()
        )
        h_next = self.market_returns.loc[t] * hplus + hplus

        assert not h_next.isnull().values.any()
        assert not u.isnull().values.any()
        return h_next, u

    def run_backtest(
        self, initial_portfolio, start_time, end_time, policy, loglevel=logging.WARNING
    ):
        """Backtest a single policy."""
        logging.basicConfig(level=loglevel)

        results = SimulationResult(
            initial_portfolio=copy.copy(initial_portfolio),
            policy=policy,
            cash_key=self.cash_key,
            simulator=self,
        )
        h = initial_portfolio

        simulation_times = self.market_returns.index[
            (self.market_returns.index >= start_time)
            & (self.market_returns.index <= end_time)
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
                risk_free_return=self.market_returns.loc[t, self.cash_key],
                exec_time=end - start,
            )

        logging.info(
            "Backtest ended, from %s to %s"
            % (simulation_times[0], simulation_times[-1])
        )
        return results

    def run_multiple_backtest(
        self,
        initial_portf,
        start_time,
        end_time,
        policies,
        loglevel=logging.WARNING,
        parallel=True,
    ):
        """Backtest multiple policies."""

        def _run_backtest(policy):
            return self.run_backtest(
                initial_portf, start_time, end_time, policy, loglevel=loglevel
            )

        num_workers = min(multiprocess.cpu_count(), len(policies))
        if parallel:
            workers = multiprocess.Pool(num_workers)
            results = workers.map(_run_backtest, policies)
            workers.close()
            return results
        else:
            return list(map(_run_backtest, policies))

    def what_if(self, time, results, alt_policies, parallel=True):
        """Run alternative policies starting from given time."""
        # TODO fix
        initial_portf = copy.copy(results.h.loc[time])
        all_times = results.h.index
        alt_results = self.run_multiple_backtest(
            initial_portf, time, all_times[-1], alt_policies, parallel
        )
        for idx, alt_result in enumerate(alt_results):
            alt_result.h.loc[time] = results.h.loc[time]
            alt_result.h.sort_index(axis=0, inplace=True)
        return alt_results

    @staticmethod
    def reduce_signal_perturb(initial_weights, delta):
        """Compute matrix of perturbed weights given initial weights."""
        perturb_weights_matrix = np.zeros((len(initial_weights), len(initial_weights)))
        for i in range(len(initial_weights)):
            perturb_weights_matrix[i, :] = initial_weights / (
                1 - delta * initial_weights[i]
            )
            perturb_weights_matrix[i, i] = (1 - delta) * initial_weights[i]
        return perturb_weights_matrix

    def attribute(
        self, true_results, policy, selector=None, delta=1, fit="linear", parallel=True
    ):
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
                alpha_sources, Wmat[idx, :]
            )
            perturb_pols.append(new_pol)
        # Simulate
        p0 = true_results.initial_portfolio
        alt_results = self.run_multiple_backtest(
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
            prob = cvx.Problem(cvx.Minimize(error), [Pmat.T @ weights == true_arr])
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
        data["residual"] = true_arr - np.asarray((weights @ Pmat).value).ravel()
        data["RMS error"] = np.asarray(
            cvx.norm(Wmat @ Pmat - Rmat, 2, axis=0).value
        ).ravel()
        data["RMS error"] /= np.sqrt(num_sources)
        return data
