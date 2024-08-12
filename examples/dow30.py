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
"""Example of long-only portfolio among DOW-30 components.

Monthly rebalance, backtest spans from the start of data of the
components of the current index (in the 60s), the other stocks
enter the backtest as times goes on. It ends today.

This uses an explicit loop to create
:class:`cvxportfolio.MultiPeriodOptimization`
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

Running the full example involves solving many (~100) back-tests, it takes
a few minutes to half an hour or longer depending on how fast your computer is
and how many processors it has. (Each back-test occupies one processor.)
"""

# You run this from the root directory of the development environment by:
# python -m examples.dow30

import os

# Uncomment the logging lines to get online information
# from the parallel backtest routines
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    import cvxportfolio as cvx

    # we have up-to-date components of a few market indexes in the
    # universes.py example script
    from .universes import DOW30

    # we use monthly resampling to make the example run faster
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

    # define combinations of hyper-parameters
    keys = [(gamma_trade, gamma_risk)
        for gamma_trade in np.array(range(10))/10
            for gamma_risk in [.5, 1, 2, 5, 10]]

    # define many policies and back-test them in parallel
    ress = sim.backtest_many([make_policy(*key) for key in keys])

    # pick the result with the largest Sharpe ratio
    print('\n\nLARGEST SHARPE RATIO')
    idx = np.argmax([el.sharpe_ratio for el in ress])

    print('gamma_trade and gamma_risk')
    print(keys[idx])

    print('result')
    print(ress[idx])

    largest_sharpe_figure = ress[idx].plot()

    # pick the result with the largest growth rate (i.e., largest profit)
    print('\n\nLARGEST GROWTH RATE')
    idx = np.argmax([el.growth_rates.mean() for el in ress])

    print('gamma_trade and gamma_risk')
    print(keys[idx])

    print('result')
    print(ress[idx])

    largest_growth_figure = ress[idx].plot()

    # also try uniform allocation, for comparison
    print('\n\nUNIFORM (1/N) ALLOCATION FOR COMPARISON')
    result_uniform = sim.backtest(cvx.Uniform())

    print('result_uniform')
    print(result_uniform)

    figure_uniform = result_uniform.plot()

    # finally, run automatic hyper-parameter optimization
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

    hyp_optimized_figure = result_hyperparameter_optimized.plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        largest_sharpe_figure.savefig('dow30_largest_sharpe_ratio.png')
        largest_growth_figure.savefig('dow30_largest_growth_rate.png')
        figure_uniform.savefig('dow30_uniform.png')
        hyp_optimized_figure.savefig('dow30_hyperparameter_optimized.png')
    else:
        plt.show()
