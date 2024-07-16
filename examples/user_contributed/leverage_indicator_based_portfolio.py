# Copyright (C) 2024 The Cvxportfolio Contributors
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
"""This is a user-contributed example, and it may not be tested.

*Work in progress.*

This example demonstrates how to use cvxportfolio to backtest a portfolio that adjusts its leverage based on a forecast indicator.

The policy implemented in this example uses a forecast indicator to determine the target leverage. If the indicator suggests
increased market risk, the policy reduces the leverage. Conversely, if the indicator suggests a favorable market, it increases
the leverage.

The example includes a custom cost model (StressModel) similar to the one in leverage_margin_portfolio_bid_ask_modelling.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.estimator import DataEstimator
from cvxportfolio.utils import set_pd_read_only


class ForecastIndicator(object):
    """A simple forecast indicator that predicts future market stress based on historical volatility.

    The indicator provides a forecast value between 0 and 1, where a higher value indicates a higher predicted market stress.
    """
    def __init__(self, lookback_period=252):
        self.lookback_period = lookback_period

    def calculate_indicator(self, returns):
        """Calculate the forecast indicator based on the volatility of the returns over the lookback period."""
        volatility = returns[-self.lookback_period:].std()
        indicator = np.clip(volatility / 0.05, 0, 1)  # Normalize and clip the indicator value
        return indicator


class StressModel(object):
    """A simple stress model that increases transaction costs (bid-ask spread) under certain conditions.

    The model calculates the bid-ask spread based on the volatility of the asset returns. If the
    volatility exceeds a specified threshold (stress_threshold), the model considers it a stressed
    market condition and increases the spread by a stress factor (stress_factor). Otherwise, the
    base spread (base_spread) is used.

    The transaction costs are assumed to be proportional to the absolute value of the trades (u).

    Parameters:
    - base_spread: The default bid-ask spread under normal market conditions (default: 0.001).
    - stress_factor: The factor by which the spread is increased under stressed conditions (default: 5).
    - stress_threshold: The volatility threshold above which the market is considered stressed (default: 0.02).
    """
    def __init__(self, base_spread=0.001, stress_factor=5, stress_threshold=0.02):
        self.base_spread = base_spread
        self.stress_factor = stress_factor
        self.stress_threshold = stress_threshold

    def get_bid_ask_spread(self, returns):
        """Calculate the bid-ask spread based on the volatility of the returns.

        If the volatility is above a certain threshold, it's considered a stress condition,
        and the spread is increased by the stress factor.
        """
        volatility = returns.std()
        is_stressed = volatility > self.stress_threshold
        spread = self.base_spread * (self.stress_factor if is_stressed else 1)
        return spread

    def simulate(self, t, u, h_plus, past_volumes,
                 past_returns, current_prices,
                 current_weights, current_portfolio_value, **kwargs):
        """Overriding the simulate function to include the stress-adjusted transaction costs."""
        spread = self.get_bid_ask_spread(past_returns.iloc[-1])
        transaction_costs = spread * np.abs(u)  # Assuming proportional to the trade size
        return transaction_costs.sum()


# Subclass of DownloadedMarketData to represent US (only) stocks in JPYEN
class ForeignCurrencyMarketData(cvx.DownloadedMarketData):
    """Represent US stocks/ETF returns, prices, and volumes in a foreign ccy.

    Supported currencies are EURO, JPYEN, GBPOUND (the currently supported
    cash keys other than USDOLLAR).

    In the future, the default MarketData servers will be able to handle
    currency conversion, using a similar mechanism as prototyped here.

    :param universe: Yahoo Finance tickers of **US assets only**.
    :type universe: iterable
    :param datasource: For this prototype only YahooFinance.
    :type datasource: str
    :param cash_key: EURO, GBPOUND, or JPYEN
    :type cash_key: str
    """
    def __init__(
            self, universe=(), datasource='YahooFinance', cash_key='JPYEN',
            *args, **kwargs):

        assert cash_key in ['JPYEN', 'EURO', 'GBPOUND']

        # creates returns, prices, volumes dataframes
        super().__init__(universe=universe, datasource=datasource,
            cash_key=cash_key, *args, **kwargs)

        rate = self._get_exchange_rate()
        rate_return = rate.pct_change().shift(-1)

        # the cash column of returns is already in cash_key
        orig_interest_rate = self.returns.iloc[:, -1]

        self.returns = (
            1 + self.returns).multiply((1 + rate_return), axis=0) - 1
        self.returns.iloc[:, -1] = orig_interest_rate
        self.returns = set_pd_read_only(self.returns)

        self.prices = set_pd_read_only(self.prices.multiply(rate, axis=0))
        self.volumes = set_pd_read_only(self.volumes.multiply(rate, axis=0))

    def _get_exchange_rate(self):
        mapping = {'JPYEN': 'JPY=X', 'EURO': 'EUR=X', 'GBPOUND': 'GBP=X'}

        # fx rate is timestamped 0:00 UTC (~midnight London)
        rate_full = cvx.YahooFinance(mapping[self.cash_key]).data

        # take close from day before rather than open, seems cleaner
        rate = rate_full.close.shift(1)

        # reindex, taking last available one
        return rate.reindex(self.returns.index, method='ffill')


class LeverageBasedOnIndicator(cvx.policies.Policy):
    """A policy that adjusts the leverage of the portfolio based on a forecast indicator."""
    def __init__(self, target_weights, forecast_indicator, max_leverage=3.0, min_leverage=1.0):
        self.target_weights = DataEstimator(target_weights, data_includes_cash=True)
        self.forecast_indicator = forecast_indicator
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

    def values_in_time(self, t, current_weights, current_portfolio_value, past_returns, **kwargs):
        # Calculate the forecast indicator
        indicator = self.forecast_indicator.calculate_indicator(past_returns)

        # Adjust the target leverage based on the indicator
        target_leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * (1 - indicator)

        # Rescale the target weights to achieve the desired leverage
        rescaled_weights = self.target_weights.current_value * target_leverage
        return rescaled_weights

# Define the target weights and initial holdings
target_weights = pd.Series({'AAPL': 0.6, 'GOOG': 0.4, 'JPYEN': -1.0})
initial_holdings = pd.Series({'AAPL': 0, 'GOOG': 0, 'JPYEN': 10000})

# Create a forecast indicator
forecast_indicator = ForecastIndicator()

# Create the market simulator with the foreign currency market data and stress model
simulator = cvx.MarketSimulator(
    market_data=ForeignCurrencyMarketData(['AAPL', 'GOOG']),
    costs=[StressModel()]
)

# Run the backtest with the leverage based on the forecast indicator
indicator_based_leverage = simulator.backtest(
    LeverageBasedOnIndicator(target_weights, forecast_indicator),
    pd.Timestamp('2010-01-01'),
    pd.Timestamp('2023-12-31'),
    h=initial_holdings
)

print('LEVERAGE BASED ON INDICATOR')
print(indicator_based_leverage)
indicator_based_leverage.plot()
plt.show()
