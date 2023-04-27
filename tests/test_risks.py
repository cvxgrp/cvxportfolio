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

    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    w_benchmark = np.random.uniform(size=N)
    w_benchmark /= sum(w_benchmark)

    risk_model = FullCovariance(rolling=50)
    risk_model.set_benchmark(w_benchmark)

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :N-1].loc[returns.index < t].iloc[-50:].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, None, None)
    assert np.isclose(cvxpy_expression.value, (w_plus.value -
                      w_benchmark)[:-1] @ should_be @ (w_plus.value - w_benchmark)[:-1])


def test_full_sigma(returns):
    N = 10
    historical_covariances = returns.iloc[:, :N-1].rolling(50).cov().dropna()
    risk_model = FullCovariance(historical_covariances)

    w_plus = cvx.Variable(N)

    risk_model.pre_evaluation(
        returns.iloc[:, :N],
        None,
        start_time=historical_covariances.index[0][0],
        end_time=None)
    risk_model.set_benchmark(pd.Series(0., returns.iloc[:, :N].columns))
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = historical_covariances.index[123][0]
    w_plus.value = np.random.randn(N)

    risk_model.values_in_time(t, None, None, None, None)

    assert np.isclose(cvxpy_expression.value, w_plus[:-1].value @
                      historical_covariances.loc[t] @ w_plus[:-1].value)



def test_full_estimated_sigma(returns):

    risk_model = FullCovariance()

    N = returns.shape[1]
    # returns.iloc[:, -1] = 0.

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :-1].loc[returns.index < t].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, returns.loc[returns.index < t], None)
    assert np.isclose(cvxpy_expression.value,
                      w_plus[:-1].value @ should_be @ w_plus[:-1].value)


def test_rolling_window_sigma(returns):

    risk_model = FullCovariance(rolling=50)

    N = returns.shape[1]
    # returns.iloc[:, -1] = 0.

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :], None, start_time=returns.index[50], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :-1].loc[returns.index < t].iloc[-(50):].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, None, None)
    assert np.isclose(cvxpy_expression.value,
                      w_plus[:-1].value @ should_be @ w_plus[:-1].value)


def test_exponential_window_sigma(returns):

    HL = 50

    risk_model = FullCovariance(halflife=HL)

    N = returns.shape[1]
    # returns.iloc[:, -1] = 0.
    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :], None, start_time=returns.index[2], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :-1].loc[returns.index < t].ewm(
        halflife=HL).cov().iloc[-(N-1):]
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, None, None)

    assert np.isclose(cvxpy_expression.value,
                      w_plus[:-1].value @ should_be @ w_plus[:-1].value)


def test_diagonal_covariance(returns):

    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    historical_variances = returns.iloc[:, :N].rolling(
        50).var().shift(1).dropna()
    risk_model = DiagonalCovariance(np.sqrt(historical_variances))

    w_plus = cvx.Variable(N)

    risk_model.pre_evaluation(
        returns,
        None,
        start_time=historical_variances.index[0],
        end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = historical_variances.index[123]
    w_plus.value = np.random.randn(N)

    risk_model.values_in_time(t, None, None, None, None)

    assert np.isclose(cvxpy_expression.value, w_plus.value @
                      np.diag(historical_variances.loc[t]) @ w_plus.value)


def test_rolling_window_diagonal_covariance(returns):

    risk_model = RollingWindowDiagonalCovariance(lookback_period=50)

    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :N], None, start_time=returns.index[51], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :N].loc[returns.index < t].iloc[-50:].cov()
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, None, None)
    assert np.isclose(cvxpy_expression.value, w_plus.value @
                      np.diag(np.diag(should_be)) @ w_plus.value)


def test_exponential_window_diagonal_covariance(returns):

    HL = 50

    risk_model = ExponentialWindowDiagonalCovariance(half_life=HL)

    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns.iloc[:, :N], None, start_time=returns.index[2], end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :N].loc[returns.index < t].ewm(
        halflife=HL).cov().iloc[-N:]
    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t, None, None, None, None)

    assert np.isclose(cvxpy_expression.value, w_plus.value @
                      np.diag(np.diag(should_be)) @ w_plus.value)


def test_low_rank_rolling_risk(returns):

    PAST = 30
    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    risk_model = LowRankRollingRisk(lookback=PAST)

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns,
        None,
        start_time=returns.index[0],
        end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')
    should_be = returns.iloc[:, :N].loc[returns.index < t].iloc[-PAST:]
    should_be = should_be.T @ should_be / PAST

    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t,
                              current_weights=None,
                              current_portfolio_value=None,
                              past_returns=returns.loc[returns.index < t],
                              past_volumes=None)

    assert np.isclose(cvxpy_expression.value,
                      w_plus.value @  should_be @ w_plus.value)


def test_RollingWindowFactorModelRisk(returns):

    PAST = 30
    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    risk_model = RollingWindowFactorModelRisk(lookback=PAST, num_factors=2)

    w_plus = cvx.Variable(N)
    risk_model.pre_evaluation(
        returns,
        None,
        start_time=returns.index[0],
        end_time=None)
    cvxpy_expression = risk_model.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    t = pd.Timestamp('2014-06-02')

    # raise Exception
    orig = returns.iloc[:, :N].loc[returns.index < t].iloc[-PAST:]
    orig = orig.T @ orig / PAST
    eigval, eigvec = np.linalg.eigh(orig)

    should_be = (eigvec[:, -2:] @ np.diag(eigval[-2:]) @ eigvec[:, -2:].T)
    should_be += np.diag(np.diag(orig) - np.diag(should_be))

    w_plus.value = np.random.randn(N)
    risk_model.values_in_time(t,
                              current_weights=None,
                              current_portfolio_value=None,
                              past_returns=returns.loc[returns.index < t],
                              past_volumes=None)

    assert np.isclose(cvxpy_expression.value,
                      w_plus.value @  should_be @ w_plus.value)


def test_worst_case_risk(returns):
    PAST = 30
    t = pd.Timestamp('2014-06-02')
    N = returns.shape[1]
    returns.iloc[:, -1] = 0.

    risk_model0 = RollingWindowFactorModelRisk(
        lookback=PAST, num_factors=2, forecast_error_kappa=0.)
    risk_model1 = RollingWindowFactorModelRisk(
        lookback=PAST, num_factors=2, forecast_error_kappa=.5)

    worst_case = WorstCaseRisk([risk_model0, risk_model1])

    w_plus = cvx.Variable(N)
    worst_case.pre_evaluation(
        returns,
        None,
        start_time=returns.index[0],
        end_time=None)
    cvxpy_expression = worst_case.compile_to_cvxpy(w_plus, None, None)
    assert cvxpy_expression.is_convex()

    cvxpy_expression0 = risk_model0.compile_to_cvxpy(w_plus, None, None)
    cvxpy_expression1 = risk_model1.compile_to_cvxpy(w_plus, None, None)

    w_plus.value = np.random.randn(N)

    worst_case.values_in_time(t,
                              current_weights=None,
                              current_portfolio_value=None,
                              past_returns=returns.loc[returns.index < t],
                              past_volumes=None)

    assert (cvxpy_expression.value == cvxpy_expression1.value)
    assert (cvxpy_expression.value > cvxpy_expression0.value)
