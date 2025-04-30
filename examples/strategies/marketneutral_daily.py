# Copyright (C) 2025 Enzo Busseti
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
"""This is a simple market neutral example strategy on US large caps.

It uses defaults returns forecast and covariance estimates, as well as cost
models. We approximate annual borrow cost on US large capitalization stocks
with a constant of the right order of magnitude as of 2025. It uses the risk
model as a constraint, for easier interpretability.

You run it from the root of the repository in the development environment by:

.. code:: bash

    python -m examples.strategies.marketneutral_daily
"""

import cvxportfolio as cvx

from ..universes import DOW30, NDX100, SP500

HYPERPAR_OPTIMIZE_START = '2000-01-01'
TARGET_LEVERAGE = 7
TARGET_ANNUAL_VOL = 0.10
ANNUAL_SHORT_FEE = 0.5 # percent

OBJECTIVE = 'sharpe_ratio'

UNIVERSE = sorted(set(DOW30 + NDX100 + SP500))

def policy(gamma_hold, gamma_trade):
    """Create fresh policy object, also return handles to hyper-parameters.

    :param gamma_hold: Holding cost term multiplier.
    :type gamma_hold: float
    :param gamma_trade: Transaction cost aversion multiplier.
    :type gamma_trade: float, optional

    :return: Policy object and dictionary mapping hyper-parameter names (which
        must match the arguments of this function) to their respective objects.
    :rtype: tuple
    """
    gamma_hold_hp = cvx.Gamma(initial_value=gamma_hold)
    gamma_trade_hp = cvx.Gamma(initial_value=gamma_trade)
    return cvx.SinglePeriodOptimization(
        cvx.ReturnsForecast()
        - gamma_hold_hp * cvx.StocksHoldingCost(short_fees=ANNUAL_SHORT_FEE)
        - gamma_trade_hp * cvx.StocksTransactionCost(),
        [
            cvx.LeverageLimit(TARGET_LEVERAGE),
            cvx.MarketNeutral(),
            cvx.DollarNeutral(),
            cvx.FullCovariance() <= cvx.AnnualizedVolatility(TARGET_ANNUAL_VOL),
        ],
        # CVXPY stuff
        ignore_dpp=True,
        solver='ECOS',
    ), {'gamma_hold': gamma_hold_hp, 'gamma_trade': gamma_trade_hp}


if __name__ == '__main__':

    RESEARCH = False

    if not RESEARCH:
        from .strategy_executor import main
        main(policy=policy, hyperparameter_opt_start=HYPERPAR_OPTIMIZE_START,
            objective=OBJECTIVE, universe=UNIVERSE, costs=(
                cvx.StocksTransactionCost(),
                cvx.StocksHoldingCost(short_fees=ANNUAL_SHORT_FEE)))

    else:
        import logging

        import matplotlib.pyplot as plt
        logging.basicConfig(level='INFO')
        RESEARCH_START = '2012-01-01'
        RESEARCH_END = '2020-01-01'

        research_sim = cvx.StockMarketSimulator(
            market_data=cvx.DownloadedMarketData(
                universe=UNIVERSE, grace_period='7d'),
            costs=(
                cvx.StocksTransactionCost(),
                cvx.StocksHoldingCost(short_fees=ANNUAL_SHORT_FEE)))

        research_policy, _ = policy(1., 1.)

        research_sim.optimize_hyperparameters(
            research_policy, start_time=RESEARCH_START,
            objective=OBJECTIVE, end_time=RESEARCH_END)

        result_opt = research_sim.backtest(
            research_policy, start_time=RESEARCH_START, end_time=RESEARCH_END)
        print('optimized')
        print(result_opt)

        result_opt.plot()

        plt.figure()
        result_opt.growth_rates.iloc[-252*4:].cumsum().plot(label='optimized')
        plt.legend()

        plt.show()
