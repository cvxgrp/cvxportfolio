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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cvx

universe = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM",  'NKE', 'MCD', 'GE', 'CVX']

# initialize the portfolio with a signle long position in AMZN
h_init = pd.Series(0., universe)
h_init["AMZN"] = 1E9
h_init['USDOLLAR'] = 0.

gamma = 0.5
kappa = 0.05
objective = cvx.ReturnsForecast() - gamma * (
    cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost(exponent=2) - cvx.StocksHoldingCost()

constraints = [cvx.MarketNeutral()] #cvx.LongOnly(),cvx.LeverageLimit(1)]

# We can impose constraints on the portfolio weights at a given time,
# the multiperiod policy will plan in advance to optimize on tcosts
constraints += [cvx.MinWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]
constraints += [cvx.MaxWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=25)

simulator = cvx.StockMarketSimulator(universe)

result = simulator.backtest(policy, start_time='2023-03-01', end_time='2023-06-01', h=h_init)

print(result)

# plot value and weights of the portfolio in time
result.plot()
