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
"""We test the Multi-Period Optimization model on a real estate portfolio.

This is an example that shows that Cvxportfolio can work as well with
different asset classes. We use the Case-Shiller index as proxy for the price
of housing units in various metropolitan areas in the USA. We impose realistic
transaction costs, which are comparable to the annual return on the asset, and
we show that multi-period optimization is useful to correctly balance
transaction cost and expected risk-adjusted return.

We present the (Cvxportfolio native) plots of the results of various
back-tests, and also create an "efficient frontier" plot obtained by
sweeping over choices of the risk aversion hyper-parameter.
"""
import os

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    import cvxportfolio as cvx

    # These are monthly time serieses of the Case-Shiller index for
    # a selection of US metropolitan areas. You can find them on the
    # website of FRED https://fred.stlouisfed.org/
    UNIVERSE = [
        'SFXRSA', # San Francisco
        'LXXRSA', # Los Angeles
        'SEXRSA', # Seattle
        'DAXRSA', # Dallas
        'SDXRSA', # San Diego
        'MIXRSA', # Miami
        'PHXRSA', # Phoenix
        'NYXRSA', # New York
        'CHXRSA', # Chicago
        'ATXRSA', # Atlanta
        'LVXRSA', # Las Vegas
        'POXRSA', # Portland
        'WDXRSA', # Washington D.C.
        'TPXRSA', # Tampa
        'CRXRSA', # Charlotte
        'MNXRSA', # Minneapolis
        'DEXRSA', # Detroit
        'CEXRSA', # Cleveland
        'DNXRSA', # Denver
        'BOXRSA', # Boston
        ]

    # we assume that the cost of transacting
    # residential real estate is about 5%
    LINEAR_TCOST = 0.05

    simulator = cvx.MarketSimulator(
        universe = UNIVERSE,
        # we enabled the default data interface to download index
        # prices from FRED
        datasource='Fred',
        costs = [cvx.TransactionCost(
            a=LINEAR_TCOST,
            b=None, # since we don't have market volumes, we can't use the
                    # market impact term of the transaction cost model
            )]
        )

    # let's see what a uniform allocation does
    result_uniform = simulator.backtest(cvx.Uniform())
    print('BACK-TEST RESULT OF UNIFORM (1/N) ALLOCATION')
    print(result_uniform)

    # plot the result
    figure_uniform = result_uniform.plot()

    # These are risk model coefficients. They don't seem to have
    # a strong effect on this example.
    NUM_FACTORS = 5
    KAPPA = 0.1

    # This is the multi-period planning horizon. We plan for 6 months
    # in the future.
    HORIZON = 6

    policies = []

    # sweep over risk aversion
    for gamma_risk in np.logspace(0, 3, 10):
        policies.append(cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast() - gamma_risk * (
            cvx.FactorModelCovariance(num_factors=NUM_FACTORS)
            + KAPPA * cvx.RiskForecastError())
                - cvx.TransactionCost(a=LINEAR_TCOST, b=None),
                    [cvx.LongOnly(applies_to_cash=True)],
                        planning_horizon=HORIZON,
                )
            )

    # run parallel back-tests for all the policies defined above
    results = simulator.backtest_many(policies)

    print('BACK-TEST RESULT OF MPO WITH HIGHEST (OUT-OF-SAMPLE) PROFIT')
    print(results[np.argmax([el.profit for el in results])])

    # back-test result with the highest profit
    top_profit_fig = results[np.argmax([el.profit for el in results])].plot()

    # multi-period optimization efficient frontier
    efficient_frontier_figure = plt.figure()
    plt.plot(
        [result.excess_returns.std() * np.sqrt(12) for result in results],
        [result.excess_returns.mean() * 12 for result in results],
        'r*-',
        label='Multi-period optimization frontier'
        )
    plt.scatter([result_uniform.excess_returns.std() * np.sqrt(12)],
        [result_uniform.excess_returns.mean() * 12],
        label='Uniform (1/n) allocation'
        )
    plt.legend()
    plt.title('Back-Test Result (Out-Of-Sample) for Real Estate Portfolio')
    plt.xlabel('Excess risk (annualized)')
    plt.ylabel('Excess return (annualized)')

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        efficient_frontier_figure.savefig('case_shiller_frontier.png')
        figure_uniform.savefig('case_shiller_uniform.png')
        top_profit_fig.savefig('case_shiller_highest_profit.png')
    else:
        plt.show()
