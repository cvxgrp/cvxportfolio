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
"""This is a simple example strategy which we run every day.

It is a variant of the ``dow30_daily`` strategy with the Nasdaq 100 universe.
It too seems to have outperformed the index etf (``QQQ``), and our benchmarks.
We will see how it performs online.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.ndx100_daily

*Edit 2024-01-11:*

    Changed the start time for hyperparameter optimization from 2020-01-01 to
    2012-01-01 and the CVXPY solver (and, 2024-01-12, the initial values of
    hyper-parameter optimization).
"""

import cvxportfolio as cvx

from ..universes import NDX100

HYPERPAR_OPTIMIZE_START = '2012-01-01'
OBJECTIVE = 'sharpe_ratio'
INITIAL_VALUES = {'gamma_risk': 20., 'gamma_trade': 1.}


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
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        benchmark=cvx.Uniform(),
        ignore_dpp=True,
        solver='CLARABEL',
    ), {'gamma_risk': gamma_risk_hp, 'gamma_trade': gamma_trade_hp}

if __name__ == '__main__':

    RESEARCH = False

    if RESEARCH:
        import matplotlib.pyplot as plt
        import pandas as pd
        INDEX_ETF = 'QQQ'

        research_sim = cvx.StockMarketSimulator(NDX100)

        result_unif = research_sim.backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print('uniform')
        print(result_unif)

        result_market = research_sim.backtest(
            cvx.MarketBenchmark(), start_time=HYPERPAR_OPTIMIZE_START)
        print('market')
        print(result_market)

        research_policy, _ = policy(**INITIAL_VALUES)

        result_etf = cvx.StockMarketSimulator([INDEX_ETF]).backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print(INDEX_ETF)
        print(result_etf)

        research_sim.optimize_hyperparameters(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START,
            objective=OBJECTIVE)

        result_opt = research_sim.backtest(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START)
        print('optimized')
        print(result_opt)

        plt.figure()
        result_opt.growth_rates.iloc[-252*4:].cumsum().plot(label='optimized')
        result_unif.growth_rates.iloc[-252*4:].cumsum().plot(label='uniform')
        result_market.growth_rates.iloc[-252*4:].cumsum().plot(label='market')
        result_etf.growth_rates.iloc[-252*4:].cumsum().plot(label='market etf')
        plt.legend()

        plt.show()

    from .strategy_executor import main
    main(policy=policy, hyperparameter_opt_start=HYPERPAR_OPTIMIZE_START,
        objective=OBJECTIVE, universe=NDX100,
        initial_values=INITIAL_VALUES)
