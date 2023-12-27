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

It is a long-only, unit leverage, allocation on the Dow Jones universe. We
use some of the simplest default settings and only optimize over two
hyper-parameters, the risk aversion and transaction cost aversion multipliers.
The code that chooses them is below.

We use a full covariance risk model, which penalizes deviations from our
market benchmark (benchmark weights proportional to last year's total
market volumes). In our research this allocation seems to have outperformed
both the uniform allocation (1/n) over this universe, the market benchmark
itself, and an index ETF which tracks the Dow Jones.

We will see how it performs online.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.dow30daily

"""

import cvxportfolio as cvx

from ..universes import DOW30

HYPERPAR_OPTIMIZE_START = '2012-01-01'

def _policy(gamma_risk, gamma_trade):
    """Create fresh policy object, also return handles to hyper-parameters."""
    gamma_risk_hp = cvx.Gamma(initial_value=gamma_risk)
    gamma_trade_hp = cvx.Gamma(initial_value=gamma_trade)
    return cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast()
        - gamma_risk_hp * cvx.FullCovariance()
        - gamma_trade_hp * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        benchmark=cvx.MarketBenchmark()
    ), gamma_risk_hp, gamma_trade_hp

def hyperparameter_optimize():
    """Optimize hyper-parameters of the policy over back-test.

    :return: Choice of gamma risk and gamma trade.
    :rtype: dict
    """
    sim = cvx.StockMarketSimulator(DOW30)#, trading_frequency='weekly')
    policy, gamma_risk_hp, gamma_trade_hp = _policy(1., 1.)
    sim.optimize_hyperparameters(
        policy, start_time=HYPERPAR_OPTIMIZE_START,
        objective='sharpe_ratio')#, objective='information_ratio')
    return {
        'gamma_risk': gamma_risk_hp.current_value,
        'gamma_trade': gamma_trade_hp.current_value,
        }

def execute_strategy(current_holdings, market_data, gamma_risk, gamma_trade):
    """Execute this strategy.

    :param current_holdings: Current holdings in dollars.
    :type current_holdings: pandas.Series
    :param market_data: Market data server.
    :type market_data: cvxportfolio.data.MarketData
    :param gamma_risk: Risk aversion multiplier
    :type gamma_risk: float
    :param gamma_trade: Transaction cost aversion multiplier.
    :type gamma_trade: float

    :return: Output of the execute method of a Cvxportfolio policy.
    :rtype: tuple
    """
    policy, _, _ = _policy(gamma_risk, gamma_trade)
    return policy.execute(h=current_holdings, market_data=market_data)


if __name__ == '__main__':

    RESEARCH = False

    if not RESEARCH:
        from .strategy_executor import main
        main(hyperparameter_optimize, execute_strategy, universe=DOW30)

    else:
        import matplotlib.pyplot as plt

        research_sim = cvx.StockMarketSimulator(DOW30)

        research_policy = cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast()
            - cvx.Gamma() * cvx.FullCovariance()
            #- cvx.Gamma() * FactorModelCovariance(num_factors=10)
            #- cvx.Gamma() * (cvx.FullCovariance()
            # + 0.05 * cvx.Gamma() * cvx.RiskForecastError())
            - cvx.Gamma() * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            benchmark=cvx.MarketBenchmark()
        )

        result_unif = research_sim.backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print('uniform')
        print(result_unif)

        research_sim.optimize_hyperparameters(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START,
            objective='sharpe_ratio')
            #objective='information_ratio')

        result_opt = research_sim.backtest(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START)
        print('optimized')
        print(result_opt)

        result_market = research_sim.backtest(
            cvx.MarketBenchmark(), start_time=HYPERPAR_OPTIMIZE_START)
        print('market')
        print(result_market)

        result_dia = cvx.StockMarketSimulator(['DIA']).backtest(
            cvx.Uniform(), start_time=HYPERPAR_OPTIMIZE_START)
        print('dia')
        print(result_dia)

        result_unif.plot()
        result_opt.plot()
        result_market.plot()
        result_dia.plot()

        plt.figure()
        result_opt.growth_rates.iloc[-252*4:].cumsum().plot(label='optimized')
        result_unif.growth_rates.iloc[-252*4:].cumsum().plot(label='uniform')
        result_market.growth_rates.iloc[-252*4:].cumsum().plot(label='market')
        result_dia.growth_rates.iloc[-252*4:].cumsum().plot(label='market etf')
        plt.legend()

        plt.show()
