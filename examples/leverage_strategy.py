# Copyright 2024 The Cvxportfolio Contributors
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
"""
This script demonstrates a leverage-adjusted fixed weights trading strategy using the cvxportfolio library.

The strategy borrows JPY to invest in US stocks with a target leverage of 3.5. The leverage is adjusted based on
predefined rules:
- If the leverage goes above 4.2, it is reduced to the target leverage.
- If the leverage goes below 2.8, it is increased to the target leverage.

The strategy also incorporates JPY interest rate and currency risk, as well as transaction costs.

The backtest is run using historical data from 2010 to 2023, and the cumulative log returns are plotted.
"""

import cvxportfolio as cvx
from cvxportfolio.utils import set_pd_read_only
from cvxportfolio.estimator import DataEstimator
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


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
        orig_interest_rate = self.returns.iloc[:,-1]

        self.returns = (
            1 + self.returns).multiply((1 + rate_return), axis=0) - 1
        self.returns.iloc[:,-1] = orig_interest_rate
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

class LeverageAdjustedFixedWeights(cvx.policies.Policy):
    def __init__(self, target_weights, max_leverage=4.2, min_leverage=2.8):#, target_leverage=3.5):#, #margin_call_leverage=6.0):
        self.target_weights = DataEstimator(target_weights, data_includes_cash=True)
        self.max_leverage = DataEstimator(max_leverage)
        self.min_leverage = DataEstimator(min_leverage)
        #self.target_leverage = DataEstimator(target_leverage)
        # self.margin_call_leverage = margin_call_leverage

    def _rescale_weights_to_target(self, weights, scale_factor):
        lev = current_leverage = sum(abs(current_weights.iloc[:-1]))


    def values_in_time(self, t, current_weights, current_portfolio_value, **kwargs):
        # Calculate the current leverage
        current_leverage = sum(abs(current_weights.iloc[:-1]))

        # If leverage is out of bounds, trade to target weights
        if (current_leverage > self.max_leverage.current_value) or (
                current_leverage < self.min_leverage.current_value):
            print(f'At time {t}, rebalancing to target weights!')
            return self.target_weights.current_value

            # # Reduce leverage to target_leverage
            # scale_factor = self.target_leverage.current_value / current_leverage
            # target_weights = current_weights.iloc[:-1] * scale_factor
            # target_weights[current_weights.index[-1]] = 1 - target_weights.iloc[:-1].sum()
        # elif current_leverage < self.min_leverage:
        #     # Increase leverage to target_leverage
        #     scale_factor = self.target_leverage / current_leverage
        #     target_weights = current_weights[:-1] * scale_factor
        #     target_weights['JPYEN'] = 1 - target_weights.iloc[:-1].sum()
        # elif current_leverage >= self.margin_call_leverage:
        #     # Reduce leverage to target_leverage (margin call)
        #     scale_factor = self.target_leverage / current_leverage
        #     target_weights = current_weights[:-1] * scale_factor
        #     target_weights['JPYEN'] = 1 - target_weights.sum()
        # else:
        #     target_weights = current_weights

        return current_weights

# Define the target weights and initial holdings
# This has leverage 3.5
target_weights = pd.Series({'AAPL': 2, 'GOOG': 1.5, 'JPYEN': -2.5})
initial_holdings = pd.Series({'AAPL': 0, 'GOOG': 0, 'JPYEN': 10000})

simulator = cvx.MarketSimulator(
    market_data = ForeignCurrencyMarketData(['AAPL', 'GOOG']),
    costs = [], # here go the costs, interest rate is already accounted
)

buy_and_hold = simulator.backtest(
    cvx.Hold(),
    pd.Timestamp('2010-01-01'),
    pd.Timestamp('2023-12-31'),
    h = target_weights * sum(initial_holdings))

print('BUY AND HOLD')
print(buy_and_hold)
buy_and_hold.plot()

rebalance_every_day = simulator.backtest(
    cvx.FixedWeights(target_weights),
    pd.Timestamp('2010-01-01'),
    pd.Timestamp('2023-12-31'),
    h = initial_holdings)

print('REBALANCE EVERY DAY')
print(rebalance_every_day)
rebalance_every_day.plot()

target_rebalance_leverage = simulator.backtest(
    LeverageAdjustedFixedWeights(target_weights),
    pd.Timestamp('2010-01-01'),
    pd.Timestamp('2023-12-31'),
    h = initial_holdings)


print('TARGET REBALANCE LEVERAGE')
print(target_rebalance_leverage)
target_rebalance_leverage.plot()





# # Create a custom market simulator with JPY as base currency
# class CustomSimulator(cvx.MarketSimulator):
#     def simulate(self, t, t_next, h, policy, past_returns, current_returns,
#                  past_volumes, current_volumes, current_prices):
#         # Apply JPY interest rate and currency risk
#         jpy_borrowed = -h['JPYEN']
#         jpy_interest_rate = 0.01  # 1% annual interest rate
#         jpy_interest_cost = jpy_borrowed * jpy_interest_rate * (t_next - t).days / 365
#         h['JPYEN'] -= jpy_interest_cost

#         # Apply transaction costs
#         transaction_cost_ratio = 0.001  # 0.1% transaction cost
#         trades = h - policy.current_value * sum(h)
#         transaction_costs = abs(trades) * transaction_cost_ratio
#         h -= transaction_costs

#         return h * (1 + current_returns), None, None, None, None

# # Create a market simulator with the custom simulator
# simulator = CustomSimulator(returns=pd.DataFrame(aapl_returns_in_jpy, columns=['AAPL']))

# # Create the leverage-adjusted fixed weights policy
# policy = LeverageAdjustedFixedWeights(target_weights)

# # Run the backtest
# results = simulator.run_backtest(
#     initial_holdings,
#     pd.Timestamp('2010-01-01'),
#     pd.Timestamp('2023-12-31'),
#     policy
# )

# # Print the backtest results
# print(results.summary())

# # Plot the cumulative log returns
# results.cum_log_returns().plot()
# plt.show()