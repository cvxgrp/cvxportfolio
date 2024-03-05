

import cvxportfolio as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# AAPL is timestamped 9:30 NY time
aapl = cvx.YahooFinance('AAPL').data

# rate is timestamped 0:00 UTC (~midnight London)
jpy_usd = cvx.YahooFinance('JPY=X').data

# take close from day before rather than open, seems cleaner
jpy_usd_rate = jpy_usd.close.shift(1)

# reindex, taking last available one
jpy_usd_rate = jpy_usd_rate.reindex(aapl.index, method='ffill')

# ccy returns, open-to-open NY time
jpy_usd_return = jpy_usd_rate.pct_change().shift(-1)

# aapl open-to-open total returns with JPY as base ccy
aapl_returns_in_jpy = (1 + aapl['return']) * (1 + jpy_usd_return) - 1

class LeverageAdjustedFixedWeights(cvx.policies.FixedWeights):
    def __init__(self, target_weights, max_leverage=4.2, min_leverage=2.8, target_leverage=3.5, margin_call_leverage=6.0):
        super().__init__(target_weights)
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.target_leverage = target_leverage
        self.margin_call_leverage = margin_call_leverage

    def values_in_time(self, current_weights, **kwargs):
        # Calculate the current leverage
        current_leverage = sum(abs(current_weights[:-1]))

        # Adjust the leverage if necessary
        if current_leverage > self.max_leverage:
            # Reduce leverage to target_leverage
            scale_factor = self.target_leverage / current_leverage
            target_weights = current_weights[:-1] * scale_factor
            target_weights['JPYEN'] = 1 - target_weights.sum()
        elif current_leverage < self.min_leverage:
            # Increase leverage to target_leverage
            scale_factor = self.target_leverage / current_leverage
            target_weights = current_weights[:-1] * scale_factor
            target_weights['JPYEN'] = 1 - target_weights.sum()
        elif current_leverage >= self.margin_call_leverage:
            # Reduce leverage to target_leverage (margin call)
            scale_factor = self.target_leverage / current_leverage
            target_weights = current_weights[:-1] * scale_factor
            target_weights['JPYEN'] = 1 - target_weights.sum()
        else:
            target_weights = self.target_weights

        return target_weights

# Define the target weights and initial holdings
target_weights = {'AAPL': 1, 'JPYEN': -0.5}
initial_holdings = {'AAPL': 0, 'JPYEN': 10000}

# Create a custom market simulator with JPY as base currency
class CustomSimulator(cvx.MarketSimulator):
    def simulate(self, t, t_next, h, policy, past_returns, current_returns,
                 past_volumes, current_volumes, current_prices):
        # Apply JPY interest rate and currency risk
        jpy_borrowed = -h['JPYEN']
        jpy_interest_rate = 0.01  # 1% annual interest rate
        jpy_interest_cost = jpy_borrowed * jpy_interest_rate * (t_next - t).days / 365
        h['JPYEN'] -= jpy_interest_cost

        # Apply transaction costs
        transaction_cost_ratio = 0.001  # 0.1% transaction cost
        trades = h - policy.current_value * sum(h)
        transaction_costs = abs(trades) * transaction_cost_ratio
        h -= transaction_costs

        return h * (1 + current_returns), None, None, None, None

# Create a market simulator with the custom simulator
simulator = CustomSimulator(returns=pd.DataFrame(aapl_returns_in_jpy, columns=['AAPL']))

# Create the leverage-adjusted fixed weights policy
policy = LeverageAdjustedFixedWeights(target_weights)

# Run the backtest
results = simulator.run_backtest(
    initial_holdings,
    start_time=pd.Timestamp('2010-01-01'),
    end_time=pd.Timestamp('2023-12-31'),
    policy=policy
)

# Print the backtest results
print(results.summary())

# Plot the cumulative log returns
results.cum_log_returns().plot()
plt.show()