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
All the rest is the same, but we optimize hyper-parameters over a shorter
period. It too seems to have outperformed the index etf (``QQQ``), and our
benchmarks. We will see how it performs online.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.ndx100_daily
"""

import cvxportfolio as cvx

from ..universes import NDX100

HYPERPAR_OPTIMIZE_START = '2020-01-01'

def _policy(gamma_risk, gamma_trade):
    """Create fresh policy object, also return handles to hyper-parameters."""
    gamma_risk_hp = cvx.Gamma(initial_value=gamma_risk)
    gamma_trade_hp = cvx.Gamma(initial_value=gamma_trade)
    return cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast()
        - gamma_risk_hp * cvx.FullCovariance()
        - gamma_trade_hp * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1)],
        benchmark=cvx.MarketBenchmark(),
        ignore_dpp=True,
    ), gamma_risk_hp, gamma_trade_hp

def hyperparameter_optimize():
    """Optimize hyper-parameters of the policy over back-test.

    :return: Choice of gamma risk and gamma trade.
    :rtype: dict
    """
    sim = cvx.StockMarketSimulator(NDX100)
    policy, gamma_risk_hp, gamma_trade_hp = _policy(1., 1.)
    sim.optimize_hyperparameters(
        policy, start_time=HYPERPAR_OPTIMIZE_START,
        objective='sharpe_ratio')
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

    from .strategy_executor import main
    main(hyperparameter_optimize, execute_strategy, universe=NDX100)
