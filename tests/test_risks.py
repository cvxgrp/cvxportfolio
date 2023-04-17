# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
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

from cvxportfolio.risks import *


def test_benchmark(returns):
    
    N = 10
    w_benchmark = np.random.uniform(size=N)
    w_benchmark /= sum(w_benchmark)
    
    risk_model = RollingWindowFullCovariance(lookback_period=50)
    risk_model.set_benchmark(w_benchmark)
    
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(returns.iloc[:, :N+1], None, start_time=returns.index[50], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:,:N].loc[returns.index < t].iloc[-50:].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t)
    assert np.isclose(cvxpy_expression.value, (w_plus.value - w_benchmark) @ should_be @ (w_plus.value - w_benchmark))
    
    


def test_full_sigma(returns):
    N = 10
    historical_covariances = returns.iloc[:, :N].rolling(50).cov().dropna()
    risk_model = FullCovariance(historical_covariances)
        
    w_plus = cvx.Variable(N)
    
    risk_model.pre_evaluation(None, None, start_time=historical_covariances.index[0][0], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = historical_covariances.index[123][0]
    w_plus.value = np.random.randn(N)
    
    risk_model.values_in_time(t)
    
    assert cvxpy_expression.value == w_plus.value @ historical_covariances.loc[t] @ w_plus.value
    

def test_rolling_window_sigma(returns):
    
    risk_model = RollingWindowFullCovariance(lookback_period=50)
    
    N = 20
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(returns.iloc[:, :N+1], None, start_time=returns.index[50], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:,:N].loc[returns.index < t].iloc[-50:].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t)
    assert np.isclose(cvxpy_expression.value, w_plus.value @ should_be @ w_plus.value)
    

def test_exponential_window_sigma(returns):
    
    HL = 50
    
    risk_model = ExponentialWindowFullCovariance(half_life=HL)
    
    N = 20
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(returns.iloc[:, :N+1], None, start_time=returns.index[2], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:,:N].loc[returns.index < t].ewm(halflife=HL).cov().iloc[-N:]
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t)
    
    assert np.isclose(cvxpy_expression.value, w_plus.value @ should_be @ w_plus.value)
    
def test_diagonal_covariance(returns):
    N = 10
    
    historical_variances = returns.iloc[:, :N].rolling(50).var().shift(1).dropna()
    risk_model = DiagonalCovariance(np.sqrt(historical_variances))
        
    w_plus = cvx.Variable(N)
    
    risk_model.pre_evaluation(None, None, start_time=historical_variances.index[0], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = historical_variances.index[123]
    w_plus.value = np.random.randn(N)
    
    risk_model.values_in_time(t)
    
    assert np.isclose(cvxpy_expression.value, w_plus.value @ np.diag(historical_variances.loc[t]) @ w_plus.value)
    
def test_rolling_window_diagonal_covariance(returns):
    
    risk_model = RollingWindowDiagonalCovariance(lookback_period=50)
    
    N = 20
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(returns.iloc[:, :N+1], None, start_time=returns.index[51], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:,:N].loc[returns.index < t].iloc[-50:].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t)
    assert np.isclose(cvxpy_expression.value, w_plus.value @ np.diag(np.diag(should_be)) @ w_plus.value)
    

def test_exponential_window_diagonal_covariance(returns):
    
    HL = 50
    
    risk_model = ExponentialWindowDiagonalCovariance(half_life=HL)
    
    N = 20
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(returns.iloc[:, :N+1], None, start_time=returns.index[2], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    
    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:,:N].loc[returns.index < t].ewm(halflife=HL).cov().iloc[-N:]
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t)
    
    assert np.isclose(cvxpy_expression.value, w_plus.value @  np.diag(np.diag(should_be)) @ w_plus.value)
    
