import cvxportfolio as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

universe = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM",  'NKE', 'MCD', 'GE', 'CVX', 'QQQ', 'SPY']


h_init = pd.Series(0., universe)
h_init["AMZN"] = 1E9
h_init['USDOLLAR'] = 0.

# w_april20 = pd.Series(1./len(universe), universe)
# w_april20['USDOLLAR'] = 0.

objective = cvx.ReturnsForecast() - 5 * (cvx.FullCovariance() + 0.05 * cvx.RiskForecastError()) - cvx.TransactionCost()  - cvx.HoldingCost()
constraints = [cvx.MarketNeutral()] #cvx.LongOnly(),cvx.LeverageLimit(1)]

constraints += [cvx.MinWeightsAtTimes(0., [pd.Timestamp('2023-04-20')])]
constraints += [cvx.MaxWeightsAtTimes(0., [pd.Timestamp('2023-04-20')])]


policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=25)



simulator = cvx.MarketSimulator(universe)

result = simulator.backtest(policy, start_time='2023-01-01', h = [h_init])

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

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())