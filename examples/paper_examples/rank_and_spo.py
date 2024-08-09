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
"""Ranking vs. SPO example.

*Work in progress.*

This is a close translation of what was done in `this notebook
<https://github.com/cvxgrp/cvxportfolio/blob/0.0.X/examples/RankAndSPO.ipynb>`_.

In fact, you can see that its results are **identical** for the Ranking policy.

The optimization policy, instead, has different performance than in the
original example: the Sharpe ratio is now higher and volatility lower, with the
same choice of hyper-parameters, which is not optimized: it was simply chosen
to match the ranking strategy's PnL. This is due to a bug in how the risk
model was handled in the early (2016) development versions of Cvxportfolio. The
risk penalization was also applied to the cash account, while the paper clearly
states that it only applies to non-cash assets. This has been fixed in these
translation scripts and is now correct in the stable versions of Cvxportfolio.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import cvxportfolio as cvx

from .common import (paper_hcost_model, paper_optimization_tcost_model,
                     paper_returns_forecast, paper_simulated_tcost_model)
from .data_risk_model import paper_market_data, paper_risk_model

# Start and end times of the back-test.
start_t = "2012-01-01"
end_t = "2016-12-31"

# Get market data.
market_data = paper_market_data()

# Define initial weights.
w_b = pd.Series(index=market_data.returns.columns, data=1)
w_b.USDOLLAR = 0.
w_b /= sum(w_b)

# Cost models.
simulated_tcost = paper_simulated_tcost_model()
simulated_hcost = paper_hcost_model()

# Market simulator.
simulator = cvx.MarketSimulator(market_data = paper_market_data(),
     costs=[simulated_tcost, simulated_hcost])

# Optimization cost models.
optimization_tcost = paper_optimization_tcost_model()
optimization_hcost = paper_hcost_model()
return_estimate = paper_returns_forecast()

factor_exposures, factor_sigma, idyosincratic = paper_risk_model()
risk_model = cvx.FactorModelCovariance(
        F=cvx.estimator.DataEstimator(
            factor_exposures, use_last_available_time=True,
            compile_parameter=True),
        d=cvx.estimator.DataEstimator(
            idyosincratic, use_last_available_time=True),
        Sigma_F=cvx.estimator.DataEstimator(
            factor_sigma, use_last_available_time=True,
            ignore_shape_check=True))

class PaperRankAndLongShort(cvx.policies.Policy):
    """Policy that reproduces the behavior of the original RankAndLongShort.

    The original implementation of this policy bought at every time-step
    the assets with highest signal, and sold the ones with lowest.

    The stable implementation instead allocates accordingly (ranking is applied
    to the weight vector, not the trade weights vector).
    """

    def __init__(self, return_forecast, num_long, num_short, target_turnover):
        self.target_turnover = target_turnover
        self.num_long = num_long
        self.num_short = num_short
        self.return_forecast = cvx.estimator.DataEstimator(return_forecast)

    def values_in_time(self, t, current_weights, **kwargs):
        """Obtain the target weight vector.

        :param t: Current timestamp.
        :type t: pd.Timestamp
        :param past_returns: Past market returns, used to get the current
            universe (its columns).
        :type past_returns: pd.Series
        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Target allocation vector.
        :rtype: pd.Series
        """
        current_signal_sorted = pd.Series(
            self.return_forecast.current_value,
            index = current_weights.index[:-1]).sort_values()
        short_trades = current_signal_sorted.index[:self.num_short]
        long_trades = current_signal_sorted.index[-self.num_long:]

        # trade weights
        z = pd.Series(0., index=current_weights.index)
        z[short_trades] = -1.0
        z[long_trades] = 1.0
        z /= sum(abs(z))
        z *= self.target_turnover * 2 # comes from our definition of TO

        # self-financing condition
        z.iloc[-1] = -np.sum(z)

        return current_weights + z


rank_and_long_short = PaperRankAndLongShort(
    return_forecast=return_estimate, num_short=10,
    num_long=10,
    target_turnover=0.005) # in the orig. example TO was off by factor of 2
result_rank = simulator.run_backtest(
    h=1e8 * w_b, start_time=start_t, end_time=end_t, policy=rank_and_long_short
)

print('RESULT RANK-AND-LONG-SHORT')
print(result_rank)


spo = cvx.SinglePeriodOpt(
    cvx.ReturnsForecast(return_estimate) - 10 * risk_model
     - 7. * optimization_tcost - 10. * optimization_hcost,
    [cvx.LeverageLimit(3)])

result_spo = simulator.run_backtest(
    h=1e8 * w_b, start_time=start_t, end_time=end_t, policy=spo
)

print('RESULT SPO')
print(result_spo)

# Plot from the paper

ax = pd.DataFrame(
    {'Rank': (result_rank.v / 1e6), 'SPO': (result_spo.v / 1e6)}).plot()

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%dM"))

plt.ylabel("Portfolio total value")
plt.show()
# plt.savefig(plotdir + "rank_vs_spo.png")


# leverage plot
result_rank.leverage.plot(figsize=(12, 6), label='Rank')
result_spo.leverage.plot(label='SPO')
plt.legend()
plt.ylabel('Leverage')
plt.show()

# example trades plot
np.abs(result_rank.u.iloc[10].sort_values()).plot(
    logy=True,  figsize=(8, 8), label='rank, sample day')
np.abs(result_spo.u.iloc[10].sort_values()).plot(
    logy=True, label='spo, sample day')
plt.legend()
plt.ylabel('abs. val. trades vector')
plt.show()
