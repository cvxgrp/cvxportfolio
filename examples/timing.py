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
"""We show the runtime of a typical single-period optimization back-test.

This is similar to what was show in :paper:`figure 7.8 of the paper
<section.7.5>`.

Many elements matter in determining how fast a back-test can be run; here
we present a few (size of risk model, choice of numerical solver and CVXPY
flags, ...) but many more are relevant and understanding of those comes only
with (deep) expertise in optimization software (and computer systems).

One interesting feature of Cvxportfolio is that it enables for automatic
caching of some expensive numerical procedures; one of them is estimation
of large covariance matrices. Here we show the execution time difference when
running the same back-test twice. The first time covariance matrices are
estimated and saved on disk, the second time they are loaded. This especially
matters when doing hyper-parameter optimization (the expensive calculation is
only done once).

Finally, we show that :class:`cvxportfolio.result.BacktestResult` does a good
job accounting for and reporting the time spent doing a back-test and in its
various components. You can expect it will do even more granular reporting in
future releases.

The results shown below are obtained on a Linux workstation with the latest
versions, at the time of writing, of most libraries.

.. note::

    To reproduce what is shown here you should make sure that the first
    time this script is run there are no covariance matrices already saved
    for the historical market data used here. If you run it from scratch, that
    is OK, but if you re-run this script it will pick up the covariance
    matrices already estimated. There is currently (Cvxportfolio ``1.3``) no
    easy way to remove caches other than manually deleting files in
    ``~/cvxportfolio_data``, which you can always safely do.
"""
import os

if __name__ == '__main__':

    import time

    import matplotlib.pyplot as plt
    import pandas as pd

    import cvxportfolio as cvx

    # same choice as in the paper
    from .universes import SP500 as UNIVERSE

    # changing these may have some effect on the solver time, but small
    GAMMA_RISK = 1.
    GAMMA_TRADE = 1.
    GAMMA_HOLD = 1.

    # the solve time grows (approximately) linearly with this. 15 is the same
    # number we had in the paper examples
    NUM_RISK_FACTORS = 15

    # if you change this to 2 (quadratic model) the resulting program is a QP
    # and can be solved faster
    TCOST_EXPONENT = 1.5

    # you can add any constraint or objective
    # term to see how it affects execution time
    policy = cvx.SinglePeriodOptimization(
        objective = cvx.ReturnsForecast()
            - GAMMA_RISK * cvx.FactorModelCovariance(
                num_factors=NUM_RISK_FACTORS)
            - GAMMA_TRADE * cvx.StocksTransactionCost(exponent=TCOST_EXPONENT)
            - GAMMA_HOLD * cvx.StocksHoldingCost(),
        constraints = [
            cvx.LeverageLimit(3),
        ],

        # You can select any CVXPY-interfaced solver here to see how it affects
        # execution time of your particular program. Different solvers apply
        # different roundings and other numerical heuristics; their solutions
        # may also have (small) differences in other back-test statistics, such
        # as Sharpe Ratio. This solver is the default one for this
        # type of programs, as of CVXPY 1.5.0. Other freely available solvers
        # that work well for this type of programs are for example CVXOPT, ECOS
        # and SCS. There are numerous commercial ones as well. See the CVXPY
        # documentation for the full list
        # https://www.cvxpy.org/tutorial/solvers/index.html
        solver='CLARABEL',

        # this is a CVXPY compilation flag, it is recommended for large
        # optimization programs (like this one) but not for small ones
        ignore_dpp=True,

        # you can add any other cvxpy.Problem.solve option
        # here, see https://www.cvxpy.org/tutorial/advanced/index.html
    )

    # this downloads data for all the sp500
    simulator = cvx.StockMarketSimulator(UNIVERSE)

    # we repeat two times to see the difference due to estimation and saving
    # of covariance matrices (the first run), and loading them from disk the
    # second time
    figures = {}
    for run in ['first', 'second']:
        # execution and timing, 5 years backtest
        s = time.time()
        result = simulator.backtest(
            policy,
            start_time=pd.Timestamp.today() - pd.Timedelta(f'{365.24*5}d'))

        print('\n\n' + run.upper() + ' RUN')

        print('BACK-TEST TOOK:', time.time() - s)
        print(
            'SIMULATOR + POLICY TIMES:',
            result.simulator_times.sum() + result.policy_times.sum())
        print(
            'AVERAGE TIME PER ITERATION:',
            result.simulator_times.mean() + result.policy_times.mean())

        print('RESULT:')
        print(result)

        # plot; this method was introduced in Cvxportfolio 1.3.0
        figures[run] = result.times_plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        figures['first'].savefig('timing_first_run.png')
        figures['second'].savefig('timing_second_run.png')
    else:
        plt.show()
