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
"""Market (and dollar) neutral strategy on the NDX100 universe.

We use standard historical means and factor covariances to build a simple
example of a market neutral strategy.

We use symbolic hyper-parameters (to be improved, see also
``examples.risk_models``) to choose the values that maximize Sharpe ratio
of the back-test, for illustrative purposes only.

We use all default values for transaction cost (both spread and market impact
terms) and holding cost (stocks borrow fees) models. These are used both for
the optimization and the back-test in
:class:`cvxportfolio.StockMarketSimulator`.

To improve the Sharpe ratio of this kind of strategies, in practice, one would
use returns forecasts produced by some machine learning model. It is very easy
to plug such forecasts into this strategy, either by providing them as a
Dataframe or by coding the forecasting logic as a Cvxportfolio native
forecaster class, and passing either as argument to
:class:`cvxportfolio.ReturnsForecast`.
"""

import os

import numpy as np

import cvxportfolio as cvx

from .universes import NDX100 as UNIVERSE

# Times.
START = '2016-01-01'
END = None # today

policy = cvx.SinglePeriodOptimization(
    objective=cvx.ReturnsForecast()
        - cvx.Gamma() * cvx.FactorModelCovariance(num_factors=10)
        - cvx.Gamma() * cvx.StocksTransactionCost()
        - cvx.Gamma() * cvx.StocksHoldingCost(),
    constraints = [
        cvx.DollarNeutral(), cvx.MarketNeutral(), cvx.LeverageLimit(7)],
    solver='CLARABEL',
    # ignore_dpp=True, #  if you increase number of factors or universe size
)

simulator = cvx.StockMarketSimulator(universe=UNIVERSE)

simulator.optimize_hyperparameters(
    policy, start_time=START, end_time=END,
    objective='sharpe_ratio')

print('Optimized policy hyper-parameters:')
print(policy)

# Back-test the policy with optimized hyper-parameters.
result = simulator.backtest(policy, start_time=START, end_time=END)

print("Optimized policy back-test result:")
print(result)

# Plot.
result.plot()

# Check that back-tested returns of the strategy are uncorrelated with the
# market benchmark.
market_benchmark_returns = simulator.backtest(
    cvx.MarketBenchmark(), start_time=START, end_time=END).returns

print('Correlation of strategy returns with benchmark:')
print(np.corrcoef(result.returns, market_benchmark_returns)[0, 1])
