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
"""This is an example of a very large backtest.

It has about ~600 names and ~6000 days, and uses multi period optimization.
It shows that all parts of the system scale to such usecases. Running it
might take a while (about half an hour on a 2020s computer),
especially the first run which computes and caches the risk model each day.
"""

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
