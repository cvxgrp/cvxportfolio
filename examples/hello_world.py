import cvxportfolio as cvx
import matplotlib.pyplot as plt

gamma = 3       # risk aversion parameter (Chapter 4.2)
kappa = 0.05    # covariance forecast error risk parameter (Chapter 4.3)
objective = cvx.ReturnsForecast() - gamma * (
	cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.TransactionCost()
constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.MarketSimulator(['AAPL', 'AMZN', 'TSLA', 'GM', 'CVX', 'NKE'])

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

