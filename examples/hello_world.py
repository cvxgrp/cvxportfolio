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
