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
""""""
import time

import matplotlib.pyplot as plt
import pandas as pd

import cvxportfolio as cvx

from .universes import SP500

# we test the typical time it takes
# to solve and simulate an SPO policy,
# in the same way as we did in Figure
# 7.8 of the book

# NOTE: the first time you run this, it will
# compute and cache the risk model terms (for
# each day). The second time you run you should
# see faster runtime.

# changing these may have some effect
# on the solver time, but small
GAMMA_RISK = 1.
GAMMA_TRADE = 1.
GAMMA_HOLD = 1.

# the solve time grows linearly
# with this. 15 is the same number
# we had in the book examples
NUM_RISK_FACTORS = 15

# if you change this to 2 (quadratic model)
# the resulting problem is a QP and can be
# solved faster
TCOST_EXPONENT = 1.5

# you can add any constraint or objective
# term to see how it affects execution time
policy = cvx.SinglePeriodOptimization(
    objective = cvx.ReturnsForecast()
        - GAMMA_RISK * cvx.FactorModelCovariance(num_factors=NUM_RISK_FACTORS)
        - GAMMA_TRADE * cvx.StocksTransactionCost(exponent=TCOST_EXPONENT)
        - GAMMA_HOLD * cvx.StocksHoldingCost(),
    constraints = [
        cvx.LeverageLimit(3),
    ],

    # You can select any CVXPY
    # solver here to see how it
    # affects performance of your
    # particular problem. This one
    # is the default for this type
    # of problems.
    solver='ECOS',

    # this is a CVXPY compilation flag, it is
    # recommended for large optimization problems
    # (like this one) but not for small ones
    ignore_dpp=True,

    # you can add any other cvxpy.Problem.solve option
    # here, see https://www.cvxpy.org/tutorial/advanced/index.html
)

# this downloads data for all the sp500
simulator = cvx.StockMarketSimulator(SP500)

# execution and timing, 5 years backtest
s = time.time()
result = simulator.backtest(
    policy, start_time=pd.Timestamp.today() - pd.Timedelta(f'{365.24*5}d'))

print('## RESULT')
print(result)

print('BACKTEST TOOK:', time.time() - s)
print(
    'SIMULATOR + POLICY TIMES:',
    result.simulator_times.sum() + result.policy_times.sum())
print(
    'AVERAGE TIME PER ITERATION:',
    result.simulator_times.mean() + result.policy_times.mean())

# plot
result.policy_times.plot(label='policy times')
result.simulator_times.plot(label='simulator times')
plt.legend()
plt.show()
