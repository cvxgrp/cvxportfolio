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
"""*Work in progress.*"""

import matplotlib.pyplot as plt
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

results = {}

## SPO coarse search
policies = {}
gamma_risks_coarse = [.1, .3, 1, 3, 10, 30, 100, 300, 1000]
gamma_tcosts_coarse = [1, 2, 5, 10, 20]
for gamma_risk in gamma_risks_coarse:
    for gamma_tcost in gamma_tcosts_coarse :
        policies[(gamma_risk, gamma_tcost)] = \
      cvx.SinglePeriodOpt(
        cvx.ReturnsForecast(return_estimate) - gamma_risk*risk_model
        - gamma_tcost*optimization_tcost
        - optimization_hcost, [cvx.LeverageLimit(3)])

results.update(dict(zip(policies.keys(),
    simulator.run_multiple_backtest(h=[1E8*w_b] * len(policies),
    start_time=start_t, end_time=end_t,
    policies=policies.values(), parallel=True))))

result_df_coarse = pd.DataFrame()
for k in results:
    if k[0] in gamma_risks_coarse and k[1] in gamma_tcosts_coarse:
        result_df_coarse.loc[k[0], k[1]] = results[k]

result_df = result_df_coarse.loc[
    sorted(result_df_coarse.index), sorted(result_df_coarse.columns)]

# plt.figure(figsize=(8,5))
for gamma_tcost in result_df.columns:
    x = [el.excess_returns.std()*100*np.sqrt(250) for el in result_df[gamma_tcost]]
    y = [el.excess_returns.mean()*100*250 for el in result_df[gamma_tcost]]
    plt.plot(np.array(x), np.array(y), '.-', label='$\gamma^\mathrm{trade} = %g$'%gamma_tcost)
plt.legend(loc='lower right')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.xlim([0, 20])
plt.ylim([0, 30])


import matplotlib.ticker as mtick

ax = plt.gca()
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

# plt.savefig(plotdir+'spo_riskrewardfrontier.png')
