# Copyright 2024 The Cvxportfolio Contributors
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
"""

@ andrew this file is to be deleted, correct?

"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import cvxportfolio as cvx

# Define a subclass of DownloadedMarketData to rebase asset returns in JPY
class JPYRebasedMarketData(cvx.DownloadedMarketData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exchange_rate_to_jpy = self.get_exchange_rate_to_jpy()

    def get_exchange_rate_to_jpy(self):
        # Dummy implementation, in practice this would fetch real exchange rate data
        # For example, it could be a pandas Series with a datetime index
        return pd.Series([1], index=pd.date_range(start='2020-01-01', periods=1))

    def returns_rebased_in_jpy(self):
        return self.returns.mul(self.exchange_rate_to_jpy, axis=0)

# Define a subclass of HoldingCost to represent holding costs in JPY
class JPYHoldingCost(cvx.HoldingCost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cash_key = 'JPYEN'

# Define a subclass of FixedWeights to represent a policy with fixed weights
class JPYFixedWeights(cvx.FixedWeights):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cash_key = 'JPYEN'

# Define the policy using FixedWeights with a dummy target weights vector
policy = JPYFixedWeights(target_weights=pd.Series({'Asset1': 0.5, 'Asset2': 0.5, 'JPYEN': 0.0}))

# Define the market simulator using the JPYRebasedMarketData
simulator = cvx.MarketSimulator(
    market_data=JPYRebasedMarketData(),
    costs=[JPYHoldingCost(borrow_costs=0.01)],
    cash_key='JPYEN'
)

# Execution and timing, 5 years backtest
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

# Plot
result.policy_times.plot(label='policy times')
result.simulator_times.plot(label='simulator times')
plt.legend()
plt.show()
