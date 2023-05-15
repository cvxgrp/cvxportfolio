import cvxportfolio as cvx
import matplotlib.pyplot as plt

objective = cvx.ReturnsForecast() - 3 * (
#cvx.FactorModelCovariance(num_factors=10) 
cvx.FullCovariance() 
+ 0.05 * cvx.RiskForecastError()) - cvx.TransactionCost()
constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=1)

simulator = cvx.MarketSimulator(sorted(set(["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM", 'NKE', 'MCD', 'GE', 'CVX', 
                              'XOM', 'MMM', 'UNH', 'HD', 'WMT', 'ORCL', 'INTC', 'JPM', 'BLK', 'BA', 'NVDA', 
                               'F', 'GS', 'AMD', 'CSCO', 'KO', 'HON', 'DIS', 
                                'V', 'ADBE', 'AMGN', 'CAT', 'BA', 'HON', 'JNJ', 'AXP', 'PG', 'JPM', 
                           'IBM', 'MRK', 'MMM', 'VZ', 'WBA', 'INTC', 'PEP', 'AVGO',
                            'COST', 'TMUS', 'CMCSA', 'TXN', 'NFLX', 'SBUX', 'GILD',
                            'ISRG', 'MDLZ', 'BKNG', 'AMAT', 'ADI', 'ADP', 'VRTX', 
                            'REGN', #'PYPL',
                             'FISV', 'LRCX'])))

result = simulator.backtest(policy, start_time='2020-01-01')

print(result)

# plot value of the portfolio in time
result.v.plot(figsize=(12, 5), label='Multi Period Optimization')
plt.ylabel('USD')
plt.yscale('log')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
result.w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())
