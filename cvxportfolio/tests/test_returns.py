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
        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus.value @ self.returns.iloc[123]))
        
        
    def test_full_returns_forecast(self):
        alpha_model = ReturnsForecast()
        cvxpy_expression = self.boilerplate(alpha_model)
        t = self.returns.index[123]
        alpha_model.values_in_time(t=t, past_returns = self.returns.loc[self.returns.index<t])
        self.w_plus.value = np.random.randn(self.N)
        myforecast = self.returns.loc[self.returns.index < t].mean()
        myforecast.iloc[-1] = self.returns.iloc[122, -1]
        assert np.isclose(cvxpy_expression.value, self.w_plus.value @ myforecast)
        
    def test_returns_forecast_error(self):

        delta = self.returns.std() / np.sqrt(len(self.returns))
        delta.iloc[-1] = 0

        error_risk = ReturnsForecastError(delta)
        cvxpy_expression = self.boilerplate(error_risk)
        error_risk.values_in_time(t='ciao', past_returns='hello')

        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        assert np.isclose(cvxpy_expression.value, np.abs(self.w_plus_minus_w_bm.value) @ delta)


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
        assert np.isclose(cvxpy_expression.value, np.abs(self.w_plus_minus_w_bm.value[:-1]) @ delta[:-1])
    
    

if __name__ == '__main__':
    unittest.main()




# def test_rolling_mean_returns_forecast(returns):
#
#     N = returns.shape[1]
#     alpha_model = ReturnsForecast(rolling=50)
#     alpha_model.pre_evaluation(returns, None, returns.index[50], None)
#     w_plus = cvx.Variable(N)
#
#     t = returns.index[123]
#     cvxpy_expression = alpha_model.compile_to_cvxpy(w_plus, None, None)
#     alpha_model.values_in_time(t, None, None, None, None)
#     w_plus.value = np.random.randn(N)
#     myforecast = returns.loc[returns.index < t].iloc[-50:].mean()
#     myforecast.iloc[-1] = returns.iloc[122, -1]
#
#     assert np.isclose(cvxpy_expression.value, w_plus.value @ myforecast)
    
    



# def test_exponential_mean_returns_forecast(returns):
#
#     N = returns.shape[1]
#     alpha_model = ReturnsForecast(halflife=25)
#     alpha_model.pre_evaluation(returns, None, returns.index[50], None)
#     w_plus = cvx.Variable(N)
#
#     t = returns.index[123]
#     cvxpy_expression = alpha_model.compile_to_cvxpy(w_plus, None, None)
#     alpha_model.values_in_time(t, None, None, None, None)
#     w_plus.value = np.random.randn(N)
#     myforecast = returns.loc[returns.index < t].ewm(
#         halflife=25).mean().iloc[-1]
#     myforecast.iloc[-1] = returns.iloc[122, -1]
#
#     assert np.isclose(cvxpy_expression.value, w_plus.value @ myforecast)

# def test_rolwin_returns_forecast_error(returns):
#
#     N = returns.shape[1]
#     error_risk = ReturnsForecastErrorRisk(rolling=20)
#     error_risk.pre_evaluation(returns, None, returns.index[50], None)
#     w_plus = cvx.Variable(N)
#
#     t = returns.index[123]
#     cvxpy_expression = error_risk.compile_to_cvxpy(w_plus, None, None)
#     error_risk.values_in_time(t, None, None, returns.loc[returns.index < t], None)
#     w_plus.value = np.random.randn(N)
#     delta = returns.loc[returns.index < t].iloc[-20:].std() / np.sqrt(20)
#     delta.iloc[-1] = 0.
#
#     assert np.isclose(cvxpy_expression.value, np.abs(w_plus.value[:-1]) @ delta[:-1])

