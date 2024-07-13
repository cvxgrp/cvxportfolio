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
"""This is a simple example strategy which we run every day."""

import cvxportfolio as cvx

from ..universes import SP500

HYPERPAR_OPTIMIZE_START = '2012-01-01'

OBJECTIVE = 'sharpe_ratio'#'profit'


def policy(gamma_trade):
    """Create fresh policy object, also return handles to hyper-parameters.

    :param gamma_trade: Transaction cost aversion multiplier.
    :type gamma_trade: float, optional

    :return: Policy object and dictionary mapping hyper-parameter names (which
        must match the arguments of this function) to their respective objects.
    :rtype: tuple
    """
    gamma_trade_hp = cvx.Gamma(initial_value=gamma_trade)
    return cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast()
        - gamma_trade_hp * cvx.StocksTransactionCost(),
        [cvx.LongOnly(), cvx.LeverageLimit(1), cvx.FullCovariance() <= (0.2)**2 / 252, cvx.MaxWeights(.05)],
        solver='CLARABEL',
        ignore_dpp=True,
        #benchmark=cvx.MarketBenchmark(),
    ), {'gamma_trade': gamma_trade_hp}


if __name__ == '__main__':

    RESEARCH = True

    if not RESEARCH:
        from .strategy_executor import main
        main(policy=policy, hyperparameter_opt_start=HYPERPAR_OPTIMIZE_START,
            objective=OBJECTIVE, universe=SP500)

    else:
        import matplotlib.pyplot as plt
        INDEX_ETF = 'SPY'

        research_sim = cvx.StockMarketSimulator(SP500)

        research_policy, _ = policy(1.)

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

        research_sim.optimize_hyperparameters(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START,
            objective='sharpe_ratio')

        result_opt = research_sim.backtest(
            research_policy, start_time=HYPERPAR_OPTIMIZE_START)
        print('optimized')
        print(result_opt)

        print(research_policy)

        result_unif.plot()
        result_opt.plot()
        result_market.plot()
        result_etf.plot()

        plt.figure()
        result_opt.growth_rates.iloc[-252*4:].cumsum().plot(label='optimized')
        result_unif.growth_rates.iloc[-252*4:].cumsum().plot(label='uniform')
        result_market.growth_rates.iloc[-252*4:].cumsum().plot(label='market')
        result_etf.growth_rates.iloc[-252*4:].cumsum().plot(label='market etf')
        plt.legend()

        plt.show()
