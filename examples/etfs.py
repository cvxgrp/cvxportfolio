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
"""ETFs example covering the main asset classes.

This uses an explicit loop to create Multi Period Optimization
policies with a grid of values for the risk term multiplier
and the transaction cost term multiplier.

All result objects are collected, and then the one with
largest Sharpe ratio, and the one with largest growth rate,
are shown.
"""

import os

import matplotlib.pyplot as plt

# Uncomment the logging lines to get online information
# from the parallel backtest routines
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    import numpy as np

    import cvxportfolio as cvx

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
        """Create policy object given hyper-parameter values.

        :param gamma_trade: Choice of the trading aversion multiplier.
        :type gamma_trade: float
        :param gamma_risk: Choice of the risk aversion multiplier.
        :type gamma_risk: float

        :returns: Policy object with given choices of hyper-parameters.
        :rtype: cvx.policies.Policy instance
        """
        return cvx.MultiPeriodOptimization(cvx.ReturnsForecast()
            - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            planning_horizon=6, solver='ECOS')

    keys = [(gamma_trade, gamma_risk)
        for gamma_trade in np.array(range(10))/10
            for gamma_risk in [.5, 1, 2, 5, 10]]
    ress = sim.backtest_many(
        [make_policy(*key) for key in keys], parallel=True)

    print('LARGEST SHARPE RATIO')
    idx = np.argmax([el.sharpe_ratio for el in ress])

    print('gamma_trade and gamma_risk')
    print(keys[idx])

    print('result')
    print(ress[idx])

    largest_sharpe_figure = ress[idx].plot()

    print('LARGEST GROWTH RATE')
    idx = np.argmax([el.growth_rates.mean() for el in ress])

    print('gamma_trade and gamma_risk')
    print(keys[idx])

    print('result')
    print(ress[idx])

    largest_growth_figure = ress[idx].plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        largest_sharpe_figure.savefig('etfs_largest_sharpe_ratio.png')
        largest_growth_figure.savefig('etfs_largest_growth_rate.png')
    else:
        plt.show()
