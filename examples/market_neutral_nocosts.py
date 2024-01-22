# Copyright 2024 Enzo Busseti
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
"""This is a simple example of a market (and dollar) neutral strategy.

It doesn't include transaction nor holding costs neither in the simulation
nor in the optimization, so its results may be unattainable in practice.

See the other example ``market_neutral.py`` for a version which does include
costs.
"""

import logging

import numpy as np

import cvxportfolio as cvx
from cvxportfolio.result import LOG_FORMAT, RECORD_LOGS
from .universes import SP500, DOW30, NDX100

logging.basicConfig(level=RECORD_LOGS, format=LOG_FORMAT)

UNIVERSE = sorted(set(DOW30 + NDX100 + SP500))

target_vol = 0.05 / np.sqrt(252) # annual std
risk = cvx.FullCovariance()
constraints = [
     risk <= target_vol**2,
     cvx.DollarNeutral(),
     cvx.MarketNeutral(),
     cvx.LeverageLimit(7),
     cvx.MaxWeights(0.05),
     cvx.MinWeights(-0.05),
 ]

policy_base = cvx.SinglePeriodOptimization(
 cvx.ReturnsForecast(),
 constraints = constraints,
 ignore_dpp=True,
)

sim = cvx.MarketSimulator(UNIVERSE)

result = sim.backtest(policy_base, start_time='2000-01-01')

print('BASE')

print(result)
result.plot()
