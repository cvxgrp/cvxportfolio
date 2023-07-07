"""Example of long-only portfolio among DOW-30 components.

Monthly rebalance, backtest spans from the start of data of the 
components of the current index (in the 60s) and the other stocks
enter the backtest as times goes on. It ends today.

This uses an explicit loop to create Multi Period Optimization
policies with a grid of values for the risk term multiplier
and the transaction cost term multiplier. (We don't use the
holding cost term because the portfolio is long-only.)

All result objects are collected, and then the one with 
largest Sharpe ratio, and the one with largest growth rate,
are shown. 
"""

# Uncomment the logging lines to get online information 
# from the parallel backtest routines

# import logging
# logging.basicConfig(level=logging.INFO)
# log=logging.getLogger('=>')

import cvxportfolio as cvx
import numpy as np


UNIVERSE = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',  'DOW', 
            'GS', 'HD', 'HON', 'IBM','INTC', 'JNJ',
            'JPM','MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'VZ', 'V', 'WBA', 'WMT']
    

sim = cvx.StockMarketSimulator(UNIVERSE, trading_frequency='monthly')

def make_policy(gamma_trade, gamma_risk):
    return cvx.MultiPeriodOptimization(cvx.ReturnsForecast() 
        - gamma_risk * cvx.FactorModelCovariance(num_factors=10) 
        - gamma_trade * cvx.StocksTransactionCost(), 
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        planning_horizon=6, solver='ECOS')

keys = [(gamma_trade, gamma_risk) for gamma_trade in np.array(range(10))/10 for gamma_risk in [.5, 1, 2, 5, 10]]
ress = sim.backtest_many([make_policy(*key) for key in keys])


print('\n\nLARGEST SHARPE RATIO')
idx = np.argmax([el.sharpe_ratio for el in ress])

print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()


print('\n\nLARGEST GROWTH RATE')
idx = np.argmax([el.growth_rates.mean() for el in ress])

print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()
