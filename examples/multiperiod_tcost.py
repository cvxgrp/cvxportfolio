import cvxportfolio as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

universe = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM",  'NKE', 'MCD', 'GE', 'CVX', 'QQQ', 'SPY']

# initialize the portfolio with a signle long position in AMZN
h_init = pd.Series(0., universe)
h_init["AMZN"] = 1E9
h_init['USDOLLAR'] = 0.

gamma = 0.5
kappa = 0.05
objective = cvx.ReturnsForecast() - gamma * (
    cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.TransactionCost(exponent=2) - cvx.HoldingCost()

constraints = [cvx.MarketNeutral()] #cvx.LongOnly(),cvx.LeverageLimit(1)]

# We can impose constraints on the portfolio weights at a given time,
# the multiperiod policy will plan in advance to optimize on tcosts
constraints += [cvx.MinWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]
constraints += [cvx.MaxWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=25)

simulator = cvx.StockMarketSimulator(universe)

result = simulator.backtest(
    policy,
    start_time='2020-01-01',
    h=[h_init]
)

print(result)

# plot value of the portfolio in time
result.v.plot(figsize=(12, 5), label='Multi Period Optimization')
plt.ylabel('USD')
plt.yscale('log')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
biggest_weights = np.abs(result.w.iloc[:, :-1]).max().sort_values().iloc[-10:].index
result.w[biggest_weights].plot()
plt.title('Largest 10 weights of the portfolio in time')
plt.show()

result.leverage.plot(); plt.show()
result.drawdown.plot(); plt.show()
