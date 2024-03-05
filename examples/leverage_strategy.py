import cvxportfolio as cp
import pandas as pd

# Define the trading universe and initial holdings
universe = ['SPY', 'TLT', 'XLE', 'XLF', 'XLK']
initial_holdings = {'SPY': 10000, 'TLT': 5000, 'XLE': 2000, 'XLF': 3000, 'XLK': 8000}

# Define the fixed weights for the assets
fixed_weights = pd.Series({'SPY': 0.4, 'TLT': 0.2, 'XLE': 0.1, 'XLF': 0.1, 'XLK': 0.2})

# Define the leverage parameters
initial_leverage = 3.5
max_leverage = 4.2
min_leverage = 2.8
margin_call_leverage = 6.0
target_leverage = 3.5

# Define the interest rate and currency risk parameters
jpy_interest_rate = 0.01  # 1% annual interest rate
jpy_usd_exchange_rate = 0.0091  # 1 JPY = 0.0091 USD

# Define the transaction costs
transaction_cost_ratio = 0.001  # 0.1% transaction cost

# Create a custom simulator
class CustomSimulator(cp.MarketSimulator):
    def simulate(self, t, t_next, h, policy, past_returns, current_returns,
                 past_volumes, current_volumes, current_prices):
        # Calculate the current leverage
        current_portfolio_value = sum(h)
        current_leverage = (current_portfolio_value + sum(h[:-1])) / current_portfolio_value

        # Adjust the leverage if necessary
        if current_leverage > max_leverage:
            # Sell assets to bring leverage down to target_leverage
            target_value = current_portfolio_value * target_leverage
            sell_value = current_portfolio_value - target_value
            sell_weights = h[:-1] / sum(h[:-1])
            sell_amounts = sell_value * sell_weights
            h[:-1] -= sell_amounts
            h[-1] += sum(sell_amounts) * (1 - transaction_cost_ratio)
        elif current_leverage < min_leverage:
            # Buy assets to bring leverage up to target_leverage
            target_value = current_portfolio_value * target_leverage
            buy_value = target_value - current_portfolio_value
            buy_weights = fixed_weights
            buy_amounts = buy_value * buy_weights
            h[:-1] += buy_amounts
            h[-1] -= sum(buy_amounts) * (1 + transaction_cost_ratio)
        
        # Check for margin call
        if current_leverage >= margin_call_leverage:
            # Sell assets until leverage is back to target_leverage
            target_value = current_portfolio_value * target_leverage
            sell_value = current_portfolio_value - target_value
            sell_weights = h[:-1] / sum(h[:-1])
            sell_amounts = sell_value * sell_weights
            h[:-1] -= sell_amounts
            h[-1] += sum(sell_amounts) * (1 - transaction_cost_ratio)
        
        # Apply JPY interest rate and currency risk
        jpy_borrowed = (current_leverage - 1) * current_portfolio_value
        jpy_interest_cost = jpy_borrowed * jpy_interest_rate * (t_next - t).days / 365
        h[-1] -= jpy_interest_cost * jpy_usd_exchange_rate
        
        return h * (1 + current_returns), None, None, None, None

# Create a market simulator
simulator = CustomSimulator(universe=universe)

# Run the backtest
results = simulator.run_backtest(
    initial_holdings,
    start_time=pd.Timestamp('2010-01-01'),
    end_time=pd.Timestamp('2023-12-31'),
    policy=cp.FixedWeights(fixed_weights)
)

# Print the backtest results
print(results.summary())