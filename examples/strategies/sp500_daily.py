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
"""This is a simple example strategy which we run every day.

It is a long-only, unit leverage, allocation on the Standard and Poor's 500
universe. It's very similar to the two strategies ``dow30_daily`` and
``ndx100_daily``, but here we also constrain the allocation to be close
to our chosen benchmark, :class:`cvxportfolio.MarketBenchmark` (allocation
proportional to last year's total market volumes in dollars).

This strategy also seems to have outperformed our benchmarks and an index ETF.
We will see how it performs online.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.sp500_daily
"""

import cvxportfolio as cvx

from ..universes import SP500

HYPERPAR_OPTIMIZE_START = '2023-01-01'

OBJECTIVE = 'sharpe_ratio'

def policy(gamma_risk, gamma_trade):
    """Create fresh policy object, also return handles to hyper-parameters.

    :param gamma_risk: Risk aversion multiplier.
    :type gamma_risk: float
    :param gamma_trade: Transaction cost aversion multiplier.
    :type gamma_trade: float, optional

    :return: Policy object and dictionary mapping hyper-parameter names (which
        must match the arguments of this function) to their respective objects.
    :rtype: tuple
    """
    gamma_risk_hp = cvx.Gamma(initial_value=gamma_risk)
    gamma_trade_hp = cvx.Gamma(initial_value=gamma_trade)
    return cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast()
        - gamma_risk_hp * cvx.FullCovariance()
        - gamma_trade_hp * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1),
            cvx.MaxBenchmarkDeviation(0.05),
            cvx.MinBenchmarkDeviation(-0.05)],
        benchmark=cvx.MarketBenchmark(),
        ignore_dpp=True,
    ), {'gamma_risk': gamma_risk_hp, 'gamma_trade': gamma_trade_hp}


if __name__ == '__main__':

    RESEARCH = False

    if RESEARCH:
        INDEX_ETF = 'SPY'

        research_sim = cvx.StockMarketSimulator(SP500)

        result_unif = research_sim.backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print('uniform')
        print(result_unif)

        result_market = research_sim.backtest(
            cvx.MarketBenchmark(), start_time=HYPERPAR_OPTIMIZE_START)
        print('market')
        print(result_market)

        result_etf = cvx.StockMarketSimulator([INDEX_ETF]).backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print(INDEX_ETF)
        print(result_etf)

    from .strategy_executor import main
    main(policy=policy, hyperparameter_opt_start=HYPERPAR_OPTIMIZE_START,
        objective=OBJECTIVE, universe=SP500, initial_values={
            'gamma_risk': 30., 'gamma_trade': 1.
        })
