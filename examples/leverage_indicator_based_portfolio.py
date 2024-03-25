"""
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
    """
    A simple forecast indicator that predicts future market stress based on historical volatility.

    The indicator provides a forecast value between 0 and 1, where a higher value indicates a higher predicted market stress.
    """
    def __init__(self, lookback_period=252):
        self.lookback_period = lookback_period

    def calculate_indicator(self, returns):
        """
        Calculate the forecast indicator based on the volatility of the returns over the lookback period.
        """
        volatility = returns[-self.lookback_period:].std()
        indicator = np.clip(volatility / 0.05, 0, 1)  # Normalize and clip the indicator value
        return indicator

class LeverageBasedOnIndicator(cvx.policies.Policy):
    """
    A policy that adjusts the leverage of the portfolio based on a forecast indicator.
    """
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
