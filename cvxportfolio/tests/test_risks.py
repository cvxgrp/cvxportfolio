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


import unittest
from pathlib import Path

import cvxpy as cvx
import numpy as np
import pandas as pd

from cvxportfolio.risks import *

USED_RETURNS = 10

class TestRisks(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        cls.returns = pd.read_csv(Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0]).iloc[:, :USED_RETURNS]
        cls.w_plus = cvx.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cvx.Variable(cls.returns.shape[1])
        cls.z = cvx.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]
        
    def boilerplate(self, model):
        model.pre_evaluation(universe=self.returns.columns, backtest_times=self.returns.index)
        return model.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)


    def test_full_sigma(self):
        historical_covariances = self.returns.rolling(50).cov(ddof=0).dropna()
        risk_model = FullCovariance(historical_covariances)
        
        cvxpy_expression = self.boilerplate(risk_model)

        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_covariances.index[123][0]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        risk_model.values_in_time(t=t, past_returns='Hello!')

        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value @
                          historical_covariances.loc[t] @ self.w_plus_minus_w_bm.value))
                               
    def test_full_estimated_sigma(self):

        risk_model = FullCovariance()
        
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        # N = returns.shape[1]
        # returns.iloc[:, -1] = 0.

        # w_plus = cvx.Variable(N)
        # risk_model.pre_evaluation(
        #     returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
        # cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
        
        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = past.iloc[:,:-1].T @ past.iloc[:,:-1] / len(past)
        
        self.w_plus_minus_w_bm.value = np.random.randn(self.N) 
        
        risk_model.values_in_time(t=t, past_returns=past)
        print(cvxpy_expression.value)
        
        print(self.w_plus_minus_w_bm.value[:-1] @ should_be @ self.w_plus_minus_w_bm.value[:-1])
        
        self.assertTrue(np.isclose(cvxpy_expression.value,
                          self.w_plus_minus_w_bm.value[:-1] @ should_be @ self.w_plus_minus_w_bm.value[:-1]))
                          
    def test_diagonal_covariance(self):

        # N = returns.shape[1]
        # returns.iloc[:, -1] = 0.

        historical_variances = self.returns.rolling(50).var().shift(1).dropna()
        risk_model = DiagonalCovariance(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]
        self.w_plus_minus_w_bm.value = np.random.randn(self.N) 

        risk_model.values_in_time(t, past_returns='hello')

        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value @
                          np.diag(historical_variances.loc[t]) @ self.w_plus_minus_w_bm.value))

    def test_full_diagonal_covariance(self):

        risk_model = DiagonalCovariance()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')
        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.
        
        risk_model.values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N) 
        
        self.assertTrue(np.isclose(cvxpy_expression.value, self.w_plus_minus_w_bm.value @
                          np.diag(should_be) @ self.w_plus_minus_w_bm.value))


    def test_forecast_error(self):

        historical_variances = (self.returns**2).rolling(50).mean().shift(1).dropna()
        
        risk_model = RiskForecastError(historical_variances)
        cvxpy_expression = self.boilerplate(risk_model)
        
        self.assertTrue(cvxpy_expression.is_convex())

        t = historical_variances.index[123]
        
        self.w_plus_minus_w_bm.value = np.random.randn(self.N) 

        risk_model.values_in_time(t, past_returns='hello')

        print(cvxpy_expression.value)
        print((np.abs(self.w_plus_minus_w_bm.value) @ np.sqrt(historical_variances.loc[t]))**2)
        self.assertTrue(np.isclose(cvxpy_expression.value, 
        
        (np.abs(self.w_plus_minus_w_bm.value) @ np.sqrt(historical_variances.loc[t]))**2
        
        ))

    def test_full_forecast_error(self):

        risk_model = RiskForecastError()
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())

        t = pd.Timestamp('2014-06-02')
        
        past = self.returns.loc[self.returns.index < t]
        should_be = (past**2).mean()
        should_be.iloc[-1] = 0.

        risk_model.values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)

        self.assertTrue(np.isclose(cvxpy_expression.value, 
        
        (np.abs(self.w_plus_minus_w_bm.value) @ np.sqrt(should_be))**2
        
        ))
        
    def test_low_rank_covariance(self):
        
        F = pd.DataFrame(np.random.randn(2, self.N), columns=self.returns.columns)
        d = pd.Series(np.random.uniform(self.N), self.returns.columns)
        risk_model = FactorModelCovariance(F=F, d=d)
        
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())
        
        risk_model.values_in_time(t=self.returns.index[12], past_returns='hello')
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        
        self.assertTrue(np.isclose(cvxpy_expression.value, 
            self.w_plus_minus_w_bm.value @ np.diag(d) @ self.w_plus_minus_w_bm.value + \
                ((F @ self.w_plus_minus_w_bm.value)**2).sum()))
                
    def test_estimated_low_rank_covariance(self):
        
        risk_model = FactorModelCovariance(normalize=False)
        
        cvxpy_expression = self.boilerplate(risk_model)
        self.assertTrue(cvxpy_expression.is_convex())
        
        t = pd.Timestamp('2014-06-02')
        
        past = self.returns.loc[self.returns.index < t]
        
        FS = past.T @ past / len(past)
        FS.iloc[:, -1] = 0.
        FS.iloc[-1, :] = 0.
        eigval, eigvec = np.linalg.eigh(FS)
        F = np.sqrt(eigval[-1]) * eigvec[:, -1]
        d = np.diag(FS) - F**2
        
        risk_model.values_in_time(t=t, past_returns=past)
        self.w_plus_minus_w_bm.value = np.random.randn(self.N)
        
        self.assertTrue(np.isclose(cvxpy_expression.value,
            self.w_plus_minus_w_bm.value @ np.diag(d) @ self.w_plus_minus_w_bm.value + \
                ((F @ self.w_plus_minus_w_bm.value)**2).sum()))
                
                
                
    def test_worst_case_risk(self):

        risk_model0 = FullCovariance()
        risk_model1 = DiagonalCovariance()
        worst_case = WorstCaseRisk([risk_model0, risk_model1])
        
        cvxpy_expression = self.boilerplate(worst_case)

        assert cvxpy_expression.is_convex()

        cvxpy_expression0 = risk_model0.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
        cvxpy_expression1 = risk_model1.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)

        self.w_plus_minus_w_bm.value = np.ones(self.N)

        t = pd.Timestamp('2014-06-02')
        
        worst_case.values_in_time(t=t, past_returns=self.returns.loc[self.returns.index < t])

        print(cvxpy_expression.value)
        print(cvxpy_expression0.value)
        print(cvxpy_expression1.value)
        assert (cvxpy_expression.value == cvxpy_expression0.value)
        assert (cvxpy_expression.value > cvxpy_expression1.value)
        
        # self.assertTrue(np.isclose(cvxpy_expression.value,
        #     self.w_plus_minus_w_bm.value @ np.diag(d) @ self.w_plus_minus_w_bm.value + \
        #         ((F @ self.w_plus_minus_w_bm.value)**2).sum()))
        
        # w_plus = cvx.Variable(N)
        # risk_model.pre_evaluation(
        #     returns,
        #     None,
        #     start_time=returns.index[0],
        #     end_time=None)
        # cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
        # assert cvxpy_expression.is_convex()
        #
        # t = pd.Timestamp('2014-06-02')
        #
        # # raise Exception
        # orig = returns.iloc[:, :N].loc[returns.index < t].iloc[-PAST:]
        # orig = orig.T @ orig / PAST
        # eigval, eigvec = np.linalg.eigh(orig)
        #
        # should_be = (eigvec[:, -2:] @ np.diag(eigval[-2:]) @ eigvec[:, -2:].T)
        # should_be += np.diag(np.diag(orig) - np.diag(should_be))
        #
        # w_plus.value = np.random.randn(N)
        # risk_model.values_in_time(t,
        #                           current_weights=None,
        #                           current_portfolio_value=None,
        #                           past_returns=returns.loc[returns.index < t],
        #                           past_volumes=None)
        #
        # assert np.isclose(cvxpy_expression.value,
        #                   w_plus.value @  should_be @ w_plus.value)



if __name__ == '__main__':
    unittest.main()

# def test_benchmark(returns):
#
#     N = returns.shape[1]
#     returns.iloc[:, -1] = 0.
#
#     w_benchmark = np.random.uniform(size=N)
#     w_benchmark /= sum(w_benchmark)
#
#     risk_model = FullCovariance(rolling=50)
#     risk_model.set_benchmark(w_benchmark)
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :N-1].loc[returns.index < t].iloc[-50:].cov()
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t, None, None, None, None)
#     assert np.isclose(cvxpy_expression.value, (w_plus.value -
#                       w_benchmark)[:-1] @ should_be @ (w_plus.value - w_benchmark)[:-1])







# def test_rolling_window_sigma(returns):
#
#     risk_model = FullCovariance(rolling=50)
#
#     N = returns.shape[1]
#     # returns.iloc[:, -1] = 0.
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :-1].loc[returns.index < t].iloc[-(50):].cov()
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t, None, None, None, None)
#     assert np.isclose(cvxpy_expression.value,
#                       w_plus[:-1].value @ should_be @ w_plus[:-1].value)
#
#
# def test_exponential_window_sigma(returns):
#
#     HL = 50
#
#     risk_model = FullCovariance(halflife=HL)
#
#     N = returns.shape[1]
#     # returns.iloc[:, -1] = 0.
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns.iloc[:, :], None, start_time=returns.index[2], end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :-1].loc[returns.index < t].ewm(
#         halflife=HL).cov().iloc[-(N-1):]
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t, None, None, None, None)
#
#     assert np.isclose(cvxpy_expression.value,
#                       w_plus[:-1].value @ should_be @ w_plus[:-1].value)



# def test_rolling_window_diagonal_covariance(returns):
#
#     risk_model = RollingWindowDiagonalCovariance(lookback_period=50)
#
#     N = returns.shape[1]
#     returns.iloc[:, -1] = 0.
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns.iloc[:, :N], None, start_time=returns.index[51], end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :N].loc[returns.index < t].iloc[-50:].cov()
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t, None, None, None, None)
#     assert np.isclose(cvxpy_expression.value, w_plus.value @
#                       np.diag(np.diag(should_be)) @ w_plus.value)
#
#
# def test_exponential_window_diagonal_covariance(returns):
#
#     HL = 50
#
#     risk_model = ExponentialWindowDiagonalCovariance(half_life=HL)
#
#     N = returns.shape[1]
#     returns.iloc[:, -1] = 0.
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns.iloc[:, :N], None, start_time=returns.index[2], end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :N].loc[returns.index < t].ewm(
#         halflife=HL).cov().iloc[-N:]
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t, None, None, None, None)
#
#     assert np.isclose(cvxpy_expression.value, w_plus.value @
#                       np.diag(np.diag(should_be)) @ w_plus.value)


# def test_low_rank_rolling_risk(returns):
#
#     PAST = 30
#     N = returns.shape[1]
#     returns.iloc[:, -1] = 0.
#
#     risk_model = LowRankRollingRisk(lookback=PAST)
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns,
#         None,
#         start_time=returns.index[0],
#         end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#     should_be = returns.iloc[:, :N].loc[returns.index < t].iloc[-PAST:]
#     should_be = should_be.T @ should_be / PAST
#
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t,
#                               current_weights=None,
#                               current_portfolio_value=None,
#                               past_returns=returns.loc[returns.index < t],
#                               past_volumes=None)
#
#     assert np.isclose(cvxpy_expression.value,
#                       w_plus.value @  should_be @ w_plus.value)


# def test_RollingWindowFactorModelRisk(returns):
#
#     PAST = 30
#     N = returns.shape[1]
#     returns.iloc[:, -1] = 0.
#
#     risk_model = RollingWindowFactorModelRisk(lookback=PAST, num_factors=2)
#
#     w_plus = cvx.Variable(N)
#     risk_model.pre_evaluation(
#         returns,
#         None,
#         start_time=returns.index[0],
#         end_time=None)
#     cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
#     assert cvxpy_expression.is_convex()
#
#     t = pd.Timestamp('2014-06-02')
#
#     # raise Exception
#     orig = returns.iloc[:, :N].loc[returns.index < t].iloc[-PAST:]
#     orig = orig.T @ orig / PAST
#     eigval, eigvec = np.linalg.eigh(orig)
#
#     should_be = (eigvec[:, -2:] @ np.diag(eigval[-2:]) @ eigvec[:, -2:].T)
#     should_be += np.diag(np.diag(orig) - np.diag(should_be))
#
#     w_plus.value = np.random.randn(N)
#     risk_model.values_in_time(t,
#                               current_weights=None,
#                               current_portfolio_value=None,
#                               past_returns=returns.loc[returns.index < t],
#                               past_volumes=None)
#
#     assert np.isclose(cvxpy_expression.value,
#                       w_plus.value @  should_be @ w_plus.value)



