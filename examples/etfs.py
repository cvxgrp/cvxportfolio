# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ETFs example covering the main asset classes.

This uses an explicit loop to create Multi Period Optimization
policies with a grid of values for the risk term multiplier
and the transaction cost term multiplier.

All result objects are collected, and then the one with
largest Sharpe ratio, and the one with largest growth rate,
are shown.
"""

import numpy as np

import cvxportfolio as cvx

# Uncomment the logging lines to get online information
# from the parallel backtest routines

# import logging
# logging.basicConfig(level=logging.INFO)
# log=logging.getLogger('=>')


UNIVERSE = [
    "QQQ", # nasdaq 100
    "SPY", # US large caps
    'EFA', # EAFE stocks
    "CWB", # convertible bonds
    "IWM", # US small caps
    "EEM", # EM stocks
    "GLD", # Gold
    'TLT', # long duration treasuries
    'HYG', # high yield bonds
    "EMB", # EM bonds (usd)
    'LQD', # investment grade bonds
    'PFF', # preferred stocks
    'VNQ', # US REITs
    'BND', # US total bond market
    'BIL', # US cash
    'TIP', # TIPS
    'DBC', # commodities
    ]


sim = cvx.StockMarketSimulator(UNIVERSE, trading_frequency='monthly')

def make_policy(gamma_trade, gamma_risk):
    return cvx.MultiPeriodOptimization(cvx.ReturnsForecast()
        - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
        - gamma_trade * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        planning_horizon=6, solver='ECOS')

keys = [(gamma_trade, gamma_risk) for gamma_trade in np.array(range(10))/10 for gamma_risk in [.5, 1, 2, 5, 10]]
ress = sim.backtest_many([make_policy(*key) for key in keys], parallel=True)


print('LARGEST SHARPE RATIO')
idx = np.argmax([el.sharpe_ratio for el in ress])

print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()


print('LARGEST GROWTH RATE')
idx = np.argmax([el.growth_rates.mean() for el in ress])


print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()
