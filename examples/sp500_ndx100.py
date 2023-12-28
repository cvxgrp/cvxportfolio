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
"""This is an example of a very large backtest.

It has about ~600 names and ~6000 days, and uses multi period optimization.
It shows that all parts of the system scale to such usecases. Running it
might take a while (about half an hour on a 2020s computer),
especially the first run which computes and caches the risk model each day.
"""

import matplotlib.pyplot as plt

import cvxportfolio as cvx

from .universes import NDX100, SP500

objective = cvx.ReturnsForecast() - .05 * cvx.ReturnsForecastError() \
     - 5 * (cvx.FactorModelCovariance(num_factors=50)
          + 0.1 * cvx.RiskForecastError()) \
     - cvx.StocksTransactionCost(exponent=2)  - cvx.StocksHoldingCost()

constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(
     objective, constraints, planning_horizon=3, ignore_dpp=True)

universe = sorted(set(SP500 + NDX100))
simulator = cvx.StockMarketSimulator(universe)

result = simulator.backtest(policy, start_time='2000-01-01', initial_value=1E9)

# print result backtest statistics
print(result)

# plot value and weights of the portfolio in time
result.plot()
