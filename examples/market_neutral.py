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
"""Market (and dollar) neutral strategy on the NDX100 universe.

We use standard historical means and factor covariances to build a simple
example of a market neutral strategy.

We use symbolic hyper-parameters (to be improved, see also
``examples.risk_models``) to choose the values that maximize Sharpe ratio
of the back-test, for illustrative purposes only.

We use realistic values for transaction cost
and holding cost (stocks borrow fees) models. These are used both for
the optimization and the back-test in
:class:`cvxportfolio.MarketSimulator`.

To improve the Sharpe ratio of this kind of strategies, in practice, one could
use returns forecasts produced by some machine learning model. It is very easy
to plug such forecasts into this strategy, either by providing them as a
Dataframe or by coding the forecasting logic as a Cvxportfolio native
forecaster class, and passing either as argument to
:class:`cvxportfolio.ReturnsForecast`.

.. note::

    Running this example may take some time, a few minutes on a modern
    workstation with a well-maintained Linux distribution, about half an hour
    (or longer!) on a laptop or a virtual machine in the cloud.
"""

# You run this from the root directory of the development environment by:
# python -m examples.market_neutral

import os

import matplotlib.pyplot as plt

# Uncomment the logging lines to get online information
# from the parallel back-test routines
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    import numpy as np

    import cvxportfolio as cvx

    from .universes import NDX100 as UNIVERSE

    # times
    START = '2016-01-01'
    END = None # today

    # Currently (~2024) shorting large cap US stocks costs about this,
    # in annualized percentages
    BORROW_FEES = 0.25

    # We set the bid-ask spreads at 5 basis points
    SPREAD = 5E-4

    # This is the b multiplier of the (3/2) power term in TransactionCost
    MARKET_IMPACT = 1.

    policy = cvx.SinglePeriodOptimization(
        objective=cvx.ReturnsForecast()
            - cvx.Gamma() * cvx.FactorModelCovariance(num_factors=10)
            - cvx.Gamma() * cvx.TransactionCost(a=SPREAD/2, b=MARKET_IMPACT)
            - cvx.Gamma() * cvx.HoldingCost(short_fees=BORROW_FEES),
        constraints = [
            cvx.DollarNeutral(), cvx.MarketNeutral(), cvx.LeverageLimit(7)],
        # this solver is somewhat more robust than ECOS, but less efficient
        solver='CLARABEL',
        # this is a CVXPY compilation flag that disables a feature that is very
        # useful (cache a semi-compiled problem) but its implementation scales
        # badly with the problem size; if you increase number of factors or
        # universe size, you may have to uncomment the next line
        # ignore_dpp=True,
    )

    simulator = cvx.MarketSimulator(
        universe=UNIVERSE,
        costs = [
            cvx.TransactionCost(a=SPREAD/2, b=MARKET_IMPACT),
            cvx.HoldingCost(short_fees=BORROW_FEES)])

    # automatic hyper-parameter optimization (by greedy grid search)
    simulator.optimize_hyperparameters(
        policy, start_time=START, end_time=END,
        objective='sharpe_ratio')

    print('Optimized policy hyper-parameters:')
    print(policy)

    # back-test the policy with optimized hyper-parameters
    result = simulator.backtest(policy, start_time=START, end_time=END)

    print("Optimized policy back-test result:")
    print(result)

    # plot
    result_figure = result.plot()

    # check that back-tested returns of the strategy are uncorrelated with the
    # market benchmark
    market_benchmark_returns = simulator.backtest(
        cvx.MarketBenchmark(), start_time=START, end_time=END).returns

    print('Correlation of strategy returns with benchmark:')
    print(np.corrcoef(result.returns, market_benchmark_returns)[0, 1])

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        result_figure.savefig('market_neutral_optimized.png')
    else:
        plt.show()
