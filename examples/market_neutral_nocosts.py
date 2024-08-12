# Copyright (C) 2024 Enzo Busseti
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
"""This is a simple example of a market (and dollar) neutral strategy.

We use a target volatility risk penalty, which is an alternative to the
standard described in the paper (where risk penalization is applied in the
objective function).

We this example for simplicity we don't include transaction nor holding costs,
neither in the simulation nor in the optimization, so its results may be
unattainable in practice. See :ref:`the market neutral example script
<Market-Neutral Portfolio>` for a version which does include transaction and
holding costs.

.. note::

    Running this may take some time. It is a single back-test and we use
    a single-threaded solver, so it won't occupy all system resources
    like parallel back-tests, or automatic hyper-parameter optimization, can.
    (You may still see some multi-threaded operations coming from CVXPY
    compilation routines, or some vectorized Numpy operation we use for
    estimating covariances.)
    The first time you run this, the covariance matrices for each day are
    estimated and factorized. They are then saved on disk (in the
    ``~/cvxportfolio_data`` folder). The second time you run it, for example
    changing some constraint, it will be noticeably faster.
"""

import os

import matplotlib.pyplot as plt

# Uncomment the logging lines to get online information from the back-test
# import logging
# logging.basicConfig(level='INFO')

if __name__ == '__main__':

    import numpy as np

    import cvxportfolio as cvx

    from .universes import DOW30, NDX100, SP500

    # these are a little more than 500 names, all large cap US stocks
    UNIVERSE = sorted(set(DOW30 + NDX100 + SP500))

    target_volatility = 0.05 / np.sqrt(252) # annual std

    # you can try different risk models (also combining some)
    # the full covariance makes this example quite expensive to run
    risk_model = cvx.FullCovariance()

    constraints = [
        risk_model <= target_volatility**2,
        cvx.DollarNeutral(),
        cvx.MarketNeutral(),
        cvx.LeverageLimit(7),
        cvx.MaxWeights(0.05),
        cvx.MinWeights(-0.05),
    ]

    policy = cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast(),
        constraints = constraints,
        # CVXPY's matrix caching can't be used here, its implementation is
        # only designed to work with small problem instances
        ignore_dpp=True,
        solver='ECOS')

    sim = cvx.MarketSimulator(UNIVERSE)

    result = sim.backtest(policy, start_time='2000-01-01')

    print('RESULT:')
    print(result)

    figure = result.plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        figure.savefig('market_neutral_nocosts.png')
    else:
        plt.show()
