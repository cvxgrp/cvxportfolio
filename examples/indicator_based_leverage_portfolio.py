import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.estimator import DataEstimator
from cvxportfolio.utils import set_pd_read_only
from cvxportfolio.indicators import MovingAverage

class LeverageAdjustedFixedWeights(cvx.policies.Policy):
    def __init__(self, target_weights, indicator, leverage_bounds):
        self.target_weights = DataEstimator(target_weights, data_includes_cash=True)
        self.indicator = indicator
        self.leverage_bounds = leverage_bounds

    def values_in_time(self, t, current_weights, current_portfolio_value, **kwargs):
        current_leverage = sum(abs(current_weights.iloc[:-1]))
        indicator_value = self.indicator.values_in_time_recursive(t=t, **kwargs)
        max_leverage = self.leverage_bounds[indicator_value]['max']
        min_leverage = self.leverage_bounds[indicator_value]['min']

        if (current_leverage > max_leverage) or (current_leverage < min_leverage):
            print(f'At time {t}, rebalancing to target weights!')
            return self.target_weights.current_value

        return current_weights

target_weights = pd.Series({'AAPL': 2, 'GOOG': 1.5, 'JPYEN': -2.5})
initial_holdings = pd.Series({'AAPL': 0, 'GOOG': 0, 'JPYEN': 10000})

moving_average_indicator = MovingAverage(window=30)
leverage_bounds = {
    'above': {'max': 5.0, 'min': 3.0},
    'below': {'max': 4.0, 'min': 2.0}
}

indicator_based_policy = LeverageAdjustedFixedWeights(
    target_weights=target_weights,
    indicator=moving_average_indicator,
    leverage_bounds=leverage_bounds
)

simulator = cvx.MarketSimulator(
    market_data=ForeignCurrencyMarketData(['AAPL', 'GOOG']),
    costs=[cvx.StressModel()]
)

indicator_based_backtest = simulator.backtest(
    indicator_based_policy,
    pd.Timestamp('2010-01-01'),
    pd.Timestamp('2023-12-31'),
    h=initial_holdings
)

print('INDICATOR BASED LEVERAGE ADJUSTMENT')
print(indicator_based_backtest)
indicator_based_backtest.plot()
plt.show()

class ForeignCurrencyMarketData(cvx.DownloadedMarketData):
    def __init__(
            self, universe=(), datasource='YahooFinance', cash_key='JPYEN',
            *args, **kwargs):

        assert cash_key in ['JPYEN', 'EURO', 'GBPOUND']
        super().__init__(universe=universe, datasource=datasource,
            cash_key=cash_key, *args, **kwargs)

        rate = self._get_exchange_rate()
        rate_return = rate.pct_change().shift(-1)
        orig_interest_rate = self.returns.iloc[:, -1]

        self.returns = (
            1 + self.returns).multiply((1 + rate_return), axis=0) - 1
        self.returns.iloc[:, -1] = orig_interest_rate
        self.returns = set_pd_read_only(self.returns)

        self.prices = set_pd_read_only(self.prices.multiply(rate, axis=0))
        self.volumes = set_pd_read_only(self.volumes.multiply(rate, axis=0))

    def _get_exchange_rate(self):
        mapping = {'JPYEN': 'JPY=X', 'EURO': 'EUR=X', 'GBPOUND': 'GBP=X'}
        rate_full = cvx.YahooFinance(mapping[self.cash_key]).data
        rate = rate_full.close.shift(1)
        return rate.reindex(self.returns.index, method='ffill')

class StressModel(object):
    def __init__(self, base_spread=0.001, stress_factor=5, stress_threshold=0.02):
        self.base_spread = base_spread
        self.stress_factor = stress_factor
        self.stress_threshold = stress_threshold

    def get_bid_ask_spread(self, returns):
        volatility = returns.std()
        is_stressed = volatility > self.stress_threshold
        spread = self.base_spread * (self.stress_factor if is_stressed else 1)
        return spread

    def simulate(self, t, u, h_plus, past_volumes,
                 past_returns, current_prices,
                 current_weights, current_portfolio_value, **kwargs):
        spread = self.get_bid_ask_spread(past_returns.iloc[-1])
        transaction_costs = spread * np.abs(u)
        return transaction_costs.sum()
