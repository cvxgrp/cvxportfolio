# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.

import pandas as pd

import cvxportfolio as cvx

UNIVERSE = [
    "AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM",  'NKE', 'MCD', 'GE', 'CVX']

# initialize the portfolio with a single long position in AMZN
h_init = pd.Series(0., UNIVERSE)
h_init["AMZN"] = 1E9
h_init['USDOLLAR'] = 0.

GAMMA = 0.5
KAPPA = 0.05
objective = cvx.ReturnsForecast() - GAMMA * (
    cvx.FullCovariance() + KAPPA * cvx.RiskForecastError()
) - cvx.StocksTransactionCost(exponent=2) - cvx.StocksHoldingCost()

constraints = [cvx.MarketNeutral()] #cvx.LongOnly(),cvx.LeverageLimit(1)]

# We can impose constraints on the portfolio weights at a given time,
# the multiperiod policy will plan in advance to optimize on tcosts
constraints += [cvx.MinWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]
constraints += [cvx.MaxWeightsAtTimes(0., [pd.Timestamp('2023-04-19')])]

policy = cvx.MultiPeriodOptimization(
    objective, constraints, planning_horizon=25)

simulator = cvx.StockMarketSimulator(UNIVERSE)

result = simulator.backtest(
    policy, start_time='2023-03-01', end_time='2023-06-01', h=h_init)

print(result)

# plot value and weights of the portfolio in time
result.plot()
