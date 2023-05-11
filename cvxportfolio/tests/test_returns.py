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

"""Unit tests for the return forecast objects."""

import unittest
from pathlib import Path

import cvxpy as cvx
import numpy as np
import pandas as pd
import numpy as np



from cvxportfolio.returns import *

class TestReturns(unittest.TestCase):
    
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
        
    def boilerplate(self, model):
        model.pre_evaluation(universe=self.returns.columns, backtest_times=self.returns.index)
        return model.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
        
        
    def test_returns_forecast(self):
        alpha_model = ReturnsForecast(self.returns)
        cvxpy_expression = self.boilerplate(alpha_model)
        alpha_model.values_in_time(t=self.returns.index[123], past_returns=None)
        self.w_plus.value = np.random.randn(self.N)
        print(cvxpy_expression.value)
        print(self.w_plus[:-1].value @ self.returns.iloc[123][:-1] + 
        ((self.w_plus[-1].value + np.sum(np.minimum(self.w_plus[:-1].value, 0.))) * self.returns.iloc[123][-1]))
        self.assertTrue(np.isclose(cvxpy_expression.value, 
            self.w_plus[:-1].value @ self.returns.iloc[123][:-1]
                + ((self.w_plus[-1].value + np.sum(np.minimum(self.w_plus[:-1].value, 0.))) * self.returns.iloc[123][-1])
            ))
        
        
    def test_full_returns_forecast(self):
        alpha_model = ReturnsForecast()
        cvxpy_expression = self.boilerplate(alpha_model)
        t = self.returns.index[123]
        alpha_model.values_in_time(t=t, past_returns = self.returns.loc[self.returns.index<t])
        self.w_plus.value = np.random.uniform(size=self.N)
        self.w_plus.value /= sum(self.w_plus.value)
        myforecast = self.returns.loc[self.returns.index < t].mean()
        myforecast.iloc[-1] = self.returns.iloc[122, -1]
        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus.value @ myforecast))
        
    def test_returns_forecast_error(self):

        delta = self.returns.std() / np.sqrt(len(self.returns))
        delta.iloc[-1] = 0

        error_risk = ReturnsForecastError(delta)
        cvxpy_expression = self.boilerplate(error_risk)
        error_risk.values_in_time(t='ciao', past_returns='hello')

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        self.assertTrue(np.isclose(cvxpy_expression.value, np.abs(self.w_plus_minus_w_bm.value) @ delta))


    def test_full_returns_forecast_error(self):

        error_risk = ReturnsForecastError()
        cvxpy_expression = self.boilerplate(error_risk)
        t = self.returns.index[123]
        past_returns = self.returns.loc[self.returns.index < t]
        
        error_risk.values_in_time(t=t, past_returns = past_returns)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        
        delta = past_returns.std() / np.sqrt(past_returns.count())

        print(cvxpy_expression.value)
        print(np.abs(self.w_plus_minus_w_bm.value[:-1]) @ delta[:-1])
        self.assertTrue(np.isclose(cvxpy_expression.value, np.abs(self.w_plus_minus_w_bm.value[:-1]) @ delta[:-1]))
    
    

if __name__ == '__main__':
    unittest.main()


