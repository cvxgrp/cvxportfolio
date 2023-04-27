# Copyright 2023- The Cvxportfolio Contributors
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

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio.simulator import MarketSimulator
from cvxportfolio.estimator import DataEstimator
from cvxportfolio import BackTest

import cvxportfolio as cp

def test_UNFINISHED_backtest(tmp_path):
    pol = cp.SinglePeriodOptimization(cp.ReturnsForecast(rolling=2000) - 
        cp.RollingWindowReturnsForecastErrorRisk(2000) -
        .5 * cp.RollingWindowFullCovariance(2000), 
        [#cp.LongOnly(), 
        cp.LeverageLimit(1)], verbose=True)
    sim = cp.MarketSimulator(['AAPL', 'MSFT'],#', 'GE', 'CVX', 'XOM', 'AMZN', 'ORCL', 'WMT', 'HD', 'DIS', 'MCD', 'NKE']
     base_location=tmp_path)
    backt = BackTest(pol, sim, pd.Timestamp('2023-01-01'), pd.Timestamp('2023-04-20'))
    
    #m = (np.log(backt.h.sum(1))).diff().mean()
    #s = (np.log(backt.h.sum(1))).diff().std()
    
    #print(np.sqrt(252) * m / s)
    
    