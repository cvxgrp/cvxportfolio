# Copyright (C) 2023-2024 Enzo Busseti
# Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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
#
## Earlier versions of this module had the following copyright and licensing
## notice, which is subsumed by the above.
##
### Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###    http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
"""Simulate periodic rebalancing policy, analyze transaction cost.

This is a close translation of what was done in `this notebook
<https://github.com/cvxgrp/cvxportfolio/blob/0.0.X/examples/PortfolioSimulation.ipynb>`_.

Note that the behavior of :class:`cvxportfolio.PeriodicRebalance` changed;
We now require an explicit iterable of rebalancing timestamps.
The original implementation, `you can see the code here
<https://github.com/cvxgrp/cvxportfolio/blob/1786671014f6cdbc539976b2e2795c02be31355d/cvxportfolio/policies.py#L158C5-L158C5>`_,
wasn't robust enough to be included in the main library. In fact, an API change
by Pandas broke it.

You run it, from the root of the development environment, by

.. code-block::

    python -m examples.paper_examples.portfolio_simulation
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import cvxportfolio as cvx

from .common import paper_hcost_model, paper_simulated_tcost_model
from .data_risk_model import paper_market_data

# Start and end times of the back-test.
start_t = "2012-01-01"
end_t = "2016-12-31"

# Get market data.
market_data = paper_market_data()

# Define benchmark weights.
w_b = pd.Series(index=market_data.returns.columns, data=1)
w_b.USDOLLAR = 0.
w_b /= sum(w_b)

# Cost models.
simulated_tcost = paper_simulated_tcost_model()
simulated_hcost = paper_hcost_model()

# Market simulator.
simulator = cvx.MarketSimulator(market_data = paper_market_data(),
     costs=[simulated_tcost, simulated_hcost])

# Define policies.

# Rebalancing times. We apply the same logic that was used in the
# development code. (It wasn't robust enough to be included in the main
# library.)
all_trading_timestamps = market_data.trading_calendar()
periodicities = ['day', 'week', 'month', 'quarter', 'year']
rebalancing_times_per_periodicity = {}
for p in periodicities:
    if p == 'week':
        # Pandas dropped 'week' from the attributes of DateTimeIndex...
        applied_to_ts = all_trading_timestamps.isocalendar().week.values
    else:
        applied_to_ts = getattr(all_trading_timestamps, p)
    is_rebalancing_ts = applied_to_ts[1:] != applied_to_ts[:-1]
    rebalancing_ts = all_trading_timestamps[1:][is_rebalancing_ts]
    rebalancing_times_per_periodicity[p] = rebalancing_ts

# For example, these are the annual rebalancing timestamps
print('Annual rebalancing timestamps:')
print(rebalancing_times_per_periodicity['year'])

policies = [
    cvx.PeriodicRebalance(
        target=w_b, rebalancing_times=rebalancing_times_per_periodicity[p])
    for p in periodicities]

policies.append(cvx.Hold())

# Run back-tests.
res = pd.DataFrame(
    index=['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual', 'Hold'])
for label, fund_val in [(r'\$100M', 1E8), (r'\$10B', 1E10)]:
    res[label] = simulator.run_multiple_backtest(
        h=[fund_val*w_b] * 6, # number of policies
        start_time=start_t, end_time=end_t,
        policies=policies, parallel=True)

# Compile results.
used_returns = market_data.returns.loc[
    (market_data.returns.index >= start_t)&(market_data.returns.index <= end_t)]
benchmark_returns = pd.Series(
    index=used_returns.index, data=np.dot(used_returns.values, w_b.values))

table = pd.DataFrame()
table[r'Active return'] = res.map(
    lambda res: 100*250*(res.returns - benchmark_returns).mean()).unstack()
table[r'Active risk'] = \
    res.map(lambda res: np.std(benchmark_returns - res.returns
        )*100*np.sqrt(250)).unstack()
table[r'Trans. costs'] =\
    res.map(lambda res: (res.costs['TcostModel']/res.v
        ).mean() * 100 * 250).unstack()
table[r'Turnover'] = \
    res.map(lambda res: res.turnover.mean()*100.*250.).unstack()

# Print latex table.
print('Latex table:')
table_print = pd.DataFrame(table, copy=True)
table_print.iloc[:, :] = table_print.iloc[:, :].applymap(lambda x: r'%.2f%%'%x )
print(table_print.to_latex(float_format='%.2f', escape=False).replace(
    '%', r'\%'))

# Plot.
plt.figure(figsize=(8, 5))
for v1 in table.index.levels[0][:]:
    x = table.loc[v1]['Trans. costs']
    y = table.loc[v1]['Active risk']
    plt.plot(np.array(x), np.array(y), 'o-', label='$%s\mathrm{%s}$'%(v1[:-1], v1[-1:]))

plt.legend(loc='upper right')
plt.xlabel('Transaction cost')
plt.ylabel('Risk')

ax = plt.gca()
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f%%'))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
plt.show()
