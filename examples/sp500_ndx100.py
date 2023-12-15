import matplotlib.pyplot as plt

import cvxportfolio as cvx

from .universes import NDX100, SP500

## This is an example of a very large backtest, ~600 names and ~6000 days
## with multi period optimization. It shows that all parts of the system scale
## to such usecases.

objective = cvx.ReturnsForecast() - .05 * cvx.ReturnsForecastError() \
     - 5 * (cvx.FactorModelCovariance(num_factors=50)
          + 0.1 * cvx.RiskForecastError()) \
     - cvx.StocksTransactionCost(exponent=2)  - cvx.StocksHoldingCost()

constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=3, ignore_dpp=True)
 
universe = sorted(set(SP500 + NDX100))
simulator = cvx.StockMarketSimulator(universe)

result = simulator.backtest(policy, start_time='2000-01-01', initial_value=1E9)

# print result backtest statistics
print(result)

# plot value and weights of the portfolio in time
result.plot() 
