import cvxportfolio as cvx
import matplotlib.pyplot as plt

objective = cvx.ReturnsForecast() - 5 * (cvx.FullCovariance() + 0.1 * cvx.RiskForecastError()) - cvx.TransactionCost()
constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.MarketSimulator(['AAPL', 'AMZN', 'MSFT', 'TSLA'])

result = simulator.backtest(policy, start_time='2020-01-01')

print(result)

# plot value of the portfolio in time
result.v.plot(figsize=(12, 5), label='Multi Period Optimization')
plt.ylabel('USD')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
result.w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())
