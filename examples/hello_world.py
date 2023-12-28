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
import os

import matplotlib.pyplot as plt

import cvxportfolio as cvx

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

simulator = cvx.StockMarketSimulator(
    ['AAPL', 'AMZN', 'UBER', 'ZM', 'CVX', 'TSLA', 'GM', 'ABNB', 'CTAS', 'GOOG'])

results = simulator.backtest_many([policy, cvx.Uniform()], start_time='2020-01-01')

# print statistics result of the backtest
print("\n# MULTI-PERIOD OPTIMIZATION\n", results[0])
print("\n# UNIFORM ALLOCATION:\n", results[1])

# plot value and weights of the portfolio in time for MPO
results[0].plot()

# we use this to save the plots for the documentation
if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('hello_world.png')

# plot value and weights of the portfolio in time for uniform
results[1].plot()

if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
    plt.savefig('hello_world_uniform.png')
else:
    plt.show()