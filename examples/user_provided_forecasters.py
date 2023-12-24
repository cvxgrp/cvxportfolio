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
"""REQUIRES cvxportfolio >= 0.4.4.

This example shows how the user can provide custom-made
predictors for expected returns and covariances,
at each point in time of the backtest. These can be
used seamlessly inside a cvxportfolio backtest routine.
"""

import cvxportfolio as cvx


# Here we define a class to forecast expected returns
class WindowMeanReturn:
    """Expected return as mean of recent window of past returns."""

    def __init__(self, window=20):
        self.window = window

    def values_in_time(self, past_returns, **kwargs):
        """This method computes the quantity of interest.

        It has many arguments, we only need to use past_returns
        in this case.

        NOTE: the last column of `past_returns` are the cash returns.
        You need to explicitely skip them otherwise the compiler will
        throw an error.
        """
        return past_returns.iloc[-self.window:, :-1].mean()


# Here we define a class to forecast covariances
class WindowCovariance:
    """Covariance computed on recent window of past returns."""

    def __init__(self, window=20):
        self.window = window

    def values_in_time(self, past_returns, **kwargs):
        """This method computes the quantity of interest.

        It has many arguments, we only need to use past_returns
        in this case.

        NOTE: the last column of `past_returns` are the cash returns.
        You need to explicitely skip them otherwise the compiler will
        throw an error.
        """
        return past_returns.iloc[-self.window:, :-1].cov()


# define the hyperparameters
WINDOWMU = 125
WINDOWSIGMA = 125
GAMMA_RISK = 5
GAMMA_TRADE = 3

# define the policy
policy = cvx.SinglePeriodOptimization(
    objective = cvx.ReturnsForecast(WindowMeanReturn(WINDOWMU))
        - GAMMA_RISK * cvx.FullCovariance(WindowCovariance(WINDOWSIGMA))
        - GAMMA_TRADE * cvx.StocksTransactionCost(),
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
    )

# define the simulator
simulator = cvx.StockMarketSimulator(['AAPL', 'GOOG', 'MSFT', 'AMZN'])

# backtest
result = simulator.backtest(policy, start_time='2020-01-01')

# show the result
print(result)
result.plot()
