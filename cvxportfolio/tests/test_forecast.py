# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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

"""Unit tests for the data and parameter estimator objects."""

import unittest
from pathlib import Path


import cvxpy as cvx
import numpy as np
import pandas as pd


from cvxportfolio.forecast import HistoricalMeanReturn, HistoricalMeanError, HistoricalVariance

class TestEstimators(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        # cls.sigma = pd.read_csv(Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
        # cls.volumes = pd.read_csv(Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
        cls.w_plus = cvx.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cvx.Variable(cls.returns.shape[1])
        cls.z = cvx.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]
    
    # def boilerplate(self, model):
    #     model.pre_evaluation(universe=self.returns.columns, backtest_times=self.returns.index)
    #     return model.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
         
    
    def test_mean_update(self):
        forecaster = HistoricalMeanReturn(lastforcash=True)
        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        
        for tidx in [50,51,52,55,56,57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index<t]
            mean = forecaster.values_in_time(t=t, past_returns=past_returns)
            print(mean)
            self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(mean[:-1], past_returns.iloc[:,:-1].mean()))
        
    
    def test_variance_update(self):
        forecaster = HistoricalVariance(addmean=False)
        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        
        for tidx in [50,51,52,55,56,57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index<t]
            var = forecaster.values_in_time(t=t, past_returns=past_returns)
            print(var)
            #self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(var, past_returns.var(ddof=0)[:-1]))    
            
    def test_meanerror_update(self):
        forecaster = HistoricalMeanError()
        
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[:20, 3:10] = np.nan
        
        for tidx in [50,51,52,55,56,57]:
            t = returns.index[tidx]
            past_returns = returns.loc[returns.index<t]
            val = forecaster.values_in_time(t=t, past_returns=past_returns)
            print(val)
            #self.assertTrue(mean[-1] == past_returns.iloc[-1,-1])
            self.assertTrue(np.allclose(val, past_returns.std(ddof=0)[:-1] / np.sqrt(past_returns.count()[:-1]) ))  
    
if __name__ == '__main__':
    unittest.main()
    