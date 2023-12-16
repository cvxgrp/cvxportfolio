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
"""Example of long-only portfolio among DOW-30 components.

Monthly rebalance, backtest spans from the start of data of the
components of the current index (in the 60s), the other stocks
enter the backtest as times goes on. It ends today.

This uses an explicit loop to create Multi Period Optimization
policies with a grid of values for the risk term multiplier
and the transaction cost term multiplier. (We don't use the
holding cost term because the portfolio is long-only.)

All result objects are collected, and then the one with
largest Sharpe ratio, and the one with largest growth rate,
are shown.

Finally, we show the effect of using symbolic hyper-parameters,
:class:`cvxportfolio.Gamma`, as multipliers of the risk and transaction
cost terms. We can optimize on those explicitely, by finding the values
that maximize some back-test metric (in this case, profit).

You can run this with this command from the parent directory:

.. code-block:: bash

    python -m examples.dow30
"""

# Uncomment the logging lines to get online information
# from the parallel backtest routines

# import logging
# logging.basicConfig(level=logging.INFO)
# log=logging.getLogger('=>')

import os

import matplotlib.pyplot as plt
import numpy as np

import cvxportfolio as cvx

from .universes import DOW30

sim = cvx.StockMarketSimulator(DOW30, trading_frequency='monthly')

def make_policy(gamma_trade, gamma_risk):
    """Build MPO policy given risk and trans. cost multipliers.

    :param gamma_trade: Transaction cost multiplier.
    :type gamma_trade: float or int
    :param gamma_risk: Risk model multiplier.
    :type gamma_risk: float or int
    :return: Multi-period optimization policy with given
        hyper-parameter values.
    :rtype: cvxportfolio.Policy
    """
    return cvx.MultiPeriodOptimization(cvx.ReturnsForecast()
        - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
        - gamma_trade * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        planning_horizon=6, solver='ECOS')

keys = [(gamma_trade, gamma_risk) for
    gamma_trade in np.array(range(10))/10 for gamma_risk in [.5, 1, 2, 5, 10]]
ress = sim.backtest_many([make_policy(*key) for key in keys])


print('\n\nLARGEST SHARPE RATIO')
idx = np.argmax([el.sharpe_ratio for el in ress])

print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()
# we use this to save plots for the documentation
if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('dow30_largest_sharpe_ratio.png')

print('\n\nLARGEST GROWTH RATE')
idx = np.argmax([el.growth_rates.mean() for el in ress])

print('gamma_trade and gamma_risk')
print(keys[idx])

print('result')
print(ress[idx])

ress[idx].plot()
if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('dow30_largest_growth_rate.png')

print('\n\nUNIFORM (1/N) ALLOCATION FOR COMPARISON')
result_uniform = sim.backtest(cvx.Uniform())

print('result_uniform')
print(result_uniform)

result_uniform.plot()
if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('dow30_uniform.png')

print('\n\nHYPER-PARAMETER OPTIMIZATION')
policy = cvx.MultiPeriodOptimization(cvx.ReturnsForecast()
        - cvx.Gamma() * cvx.FactorModelCovariance(num_factors=10)
        - cvx.Gamma() * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        planning_horizon=6, solver='ECOS')
sim.optimize_hyperparameters(policy, objective='profit')
result_hyperparameter_optimized = sim.backtest(policy)

print('result_hyperparameter_optimized')
print(result_hyperparameter_optimized)

result_hyperparameter_optimized.plot()

if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('dow30_hyperparameter_optimized.png')
else:
    plt.show()
