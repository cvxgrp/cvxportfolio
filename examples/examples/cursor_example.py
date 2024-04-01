


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxportfolio as cvx

class MomentumIndicator:
    """
    A simple momentum indicator that calculates the momentum of assets based on their past returns.
    The indicator provides a momentum value for each asset, where a positive value indicates positive momentum
    and a negative value indicates negative momentum.
    """
    def __init__(self, lookback_period=90):
        self.lookback_period = lookback_period
    
    def calculate_momentum(self, returns):
        """Calculate the momentum indicator for each asset based on the returns over the lookback period."""
        momentum = returns.iloc[-self.lookback_period:].mean()
        return momentum

class DynamicAllocationPolicy(cvx.Policy):
    """A policy that adjusts the asset allocation of the portfolio based on a momentum indicator."""
    def __init__(self, base_weights, momentum_indicator, max_shift=0.2):
        self.base_weights = base_weights
        self.momentum_indicator = momentum_indicator
        self.max_shift = max_shift
    
    def get_trades(self, t, h):
        # Calculate the momentum indicator for each asset
        momentum = self.momentum_indicator.calculate_momentum(self.returns)
        
        # Adjust the target weights based on the momentum
        adjusted_weights = self.base_weights.copy()
        for asset, mom in momentum.items():
            shift = np.clip(mom, -self.max_shift, self.max_shift)
            adjusted_weights[asset] += shift
        
        # Normalize the adjusted weights to ensure they sum to 1
        adjusted_weights /= adjusted_weights.sum()
        
        return adjusted_weights - h

# Define the base target weights and initial holdings
base_weights = pd.Series({'AAPL': 0.5, 'GOOG': 0.5})
initial_holdings = pd.Series({'AAPL': 100, 'GOOG': 100})

# Create a momentum indicator
momentum_indicator = MomentumIndicator()

# Create the market simulator with the default market data
simulator = cvx.MarketSimulator(
    market_data=cvx.DownloadedMarketData(['AAPL', 'GOOG'])
)

# Run the backtest with the dynamic allocation based on the momentum indicator
momentum_based_allocation = simulator.run(
    DynamicAllocationPolicy(base_weights, momentum_indicator),
    initial_holdings,
    start_time=pd.Timestamp('2010-01-01'),
    end_time=pd.Timestamp('2023-12-31')
)

print('MOMENTUM BASED ALLOCATION')
print(momentum_based_allocation)
momentum_based_allocation.plot()
plt.show()
