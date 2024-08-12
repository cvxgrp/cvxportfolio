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
"""This is a simple example of Cvxportfolio's capabilities.

A multi-period optimization policy, with default forecasts and simple
choice of objective terms and constraints, is compared to a uniform (1/n)
allocation for a certain selection of stocks and time period.

The results are printed and plotted using the default methods.
"""
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':
    import cvxportfolio as cvx

    # risk aversion parameter (Chapter 4.2)
    # chosen to match resulting volatility with the
    # uniform portfolio (for illustrative purpose)
    GAMMA = 2.5

    # covariance forecast error risk parameter (Chapter 4.3)
    # this can help regularize a noisy covariance estimate
    KAPPA = 0.05

    objective = cvx.ReturnsForecast() - GAMMA * (
        cvx.FullCovariance() + KAPPA * cvx.RiskForecastError()
    ) - cvx.StocksTransactionCost()

    constraints = [cvx.LeverageLimit(3)]

    policy = cvx.MultiPeriodOptimization(
        objective, constraints, planning_horizon=2)

    simulator = cvx.StockMarketSimulator(
        ['AAPL', 'AMZN', 'UBER', 'ZM', 'CVX', 'TSLA', 'GM', 'ABNB', 'CTAS',
        'GOOG'])

    results = simulator.backtest_many(
        [policy, cvx.Uniform()], start_time='2020-01-01')

    # print multi-period result
    print("\n# MULTI-PERIOD OPTIMIZATION\n")
    print(results[0])

    # print uniform allocation result
    print("\n# UNIFORM ALLOCATION:\n")
    print(results[1])

    # plot value and weights of the portfolio in time for MPO
    mpo_figure = results[0].plot()

    # plot value and weights of the portfolio in time for uniform
    uniform_figure = results[1].plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        mpo_figure.savefig('hello_world.png')
        uniform_figure.savefig('hello_world_uniform.png')
    else:
        plt.show()
