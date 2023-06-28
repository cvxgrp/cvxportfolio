import cvxportfolio as cvx
import matplotlib.pyplot as plt

# risk aversion parameter (Chapter 4.2)
# chosen to match resulting volatility with the
# uniform portfolio (for illustrative purpose)
gamma = 2.5 

# covariance forecast error risk parameter (Chapter 4.3)
# this can help regularize a noisy covariance estimate
kappa = 0.05  

objective = cvx.ReturnsForecast() - gamma * (
	cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost()

constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.StockMarketSimulator(['AAPL', 'AMZN', 'UBER', 'ZM', 'CVX', 'TSLA', 'GM', 'ABNB', 'CTAS', 'GOOG'])

results = simulator.backtest_many([policy, cvx.Uniform()], start_time='2020-01-01')

# print statistics result of the backtest
print("\n# MULTI-PERIOD OPTIMIZATION\n", results[0])
print("\n# UNIFORM ALLOCATION:\n", results[1])

# plot value and weights of the portfolio in time for MPO
results[0].plot() 
