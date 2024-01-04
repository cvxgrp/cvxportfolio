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

It is a long-only, unit leverage, allocation on the FTSE 100 universe.

We will see how it performs online.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.ftse100_daily
"""

import cvxportfolio as cvx

from ..universes import FTSE100

HYPERPAR_OPTIMIZE_START = '2012-01-01'

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
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        benchmark=cvx.MarketBenchmark(),
    ), {'gamma_risk': gamma_risk_hp, 'gamma_trade': gamma_trade_hp}


if __name__ == '__main__':

    RESEARCH = True

    if not RESEARCH:
        from .strategy_executor import main
        main(policy=policy, hyperparameter_opt_start=HYPERPAR_OPTIMIZE_START,
            objective=OBJECTIVE, universe=FTSE100, cash_key='GBPOUND')

    else:
        import matplotlib.pyplot as plt
        #INDEX_ETF = 'DIA'

        research_sim = cvx.StockMarketSimulator(FTSE100, cash_key='GBPOUND')

        research_policy, _ = policy(1., 1.)

        result_unif = research_sim.backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print('uniform')
        print(result_unif)

        result_market = research_sim.backtest(
            cvx.MarketBenchmark(), start_time=HYPERPAR_OPTIMIZE_START)
        print('market')
        print(result_market)

        exit(0)

        # result_etf = cvx.StockMarketSimulator([INDEX_ETF]).backtest(
        #     cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        # print(INDEX_ETF)
        # print(result_etf)

        research_sim.optimize_hyperparameters(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START,
            objective='sharpe_ratio')

        result_opt = research_sim.backtest(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START)
        print('optimized')
        print(result_opt)

        result_unif.plot()
        result_opt.plot()
        result_market.plot()
        #result_etf.plot()

        plt.figure()
        result_opt.growth_rates.iloc[-252*4:].cumsum().plot(label='optimized')
        result_unif.growth_rates.iloc[-252*4:].cumsum().plot(label='uniform')
        result_market.growth_rates.iloc[-252*4:].cumsum().plot(label='market')
        #result_etf.growth_rates.iloc[-252*4:].cumsum().plot(label='market etf')
        plt.legend()

        plt.show()
