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

from cvxportfolio.policies import *
from cvxportfolio.policies import SinglePeriodOptOLD, SinglePeriodOptNEW
from cvxportfolio.returns import *
from cvxportfolio.risks import *
from cvxportfolio.costs import *
from cvxportfolio.constraints import *
from cvxportfolio.errors import *


def test_hold():
    hold = Hold()
    w = pd.Series(0.5, ["AAPL", "CASH"])
    assert np.all(
        hold.values_in_time(
            None,
            w,
            "extra_args",
            hello="ciao").values == np.zeros(2))


def test_rank_and_long_short():
    hold = Hold()
    w = pd.Series(0.25, ["AAPL", "TSLA", "GOOGL", "CASH"])
    signal = pd.Series([1, 2, 3], ["AAPL", "TSLA", "GOOGL"])
    num_long = 1
    num_short = 1
    target_leverage = 3.0
    rls = RankAndLongShort(
        signal=signal,
        num_long=num_long,
        num_short=num_short,
        target_leverage=target_leverage,
    )
    z = rls.values_in_time(None, w, None, None, None)
    print(z)
    wplus = w + z
    assert wplus["CASH"] == 1
    assert wplus["TSLA"] == 0
    assert wplus["AAPL"] == -wplus["GOOGL"]
    assert np.abs(wplus[:-1]).sum() == 3

    index = pd.date_range("2020-01-01", "2020-01-03")
    signal = pd.DataFrame(
        {
            "AAPL": pd.Series([1, 1.9, 3], index),
            "TSLA": pd.Series([3, 2.1, 1], index),
            "GOOGL": pd.Series([4, 4, 4], index),
        }
    )
    rls = RankAndLongShort(
        signal=signal,
        num_long=num_long,
        num_short=num_short,
        target_leverage=target_leverage,
    )
    z1 = rls.values_in_time(index[0], w, None, None, None)
    print(z1)
    wplus = w + z1
    assert wplus["CASH"] == 1
    assert wplus["TSLA"] == 0
    assert wplus["AAPL"] == -wplus["GOOGL"]
    assert np.abs(wplus[:-1]).sum() == 3
    z2 = rls.values_in_time(index[1], w, None, None, None)
    print(z2)
    wplus = w + z2
    assert wplus["CASH"] == 1
    assert wplus["TSLA"] == 0
    assert wplus["AAPL"] == -wplus["GOOGL"]
    assert np.abs(wplus[:-1]).sum() == 3
    z3 = rls.values_in_time(index[2], w, None, None, None)
    wplus = w + z3
    assert wplus["CASH"] == 1
    assert wplus["AAPL"] == 0
    assert wplus["TSLA"] == -wplus["GOOGL"]
    assert np.abs(wplus[:-1]).sum() == 3
    print(z3)

    # raise Exception


def test_proportional_trade(returns):
    
    a = pd.Series(1., returns.columns)
    a[-1] = 1 - sum(a[:-1])
    b = pd.Series(-1., returns.columns)
    b[-1] = 1 - sum(b[:-1])

    targets = pd.DataFrame({returns.index[3]: a,
                            returns.index[15]: b
                            }).T
    policy = ProportionalTradeToTargets(targets)

    policy.pre_evaluation(returns, None, None, None)
    start_portfolio = pd.Series(
        np.random.randn(
            returns.shape[1]),
        returns.columns)
    start_portfolio[-1] = 1 - sum(start_portfolio[:-1])
    for t in returns.index[:17]:
        print(t)
        print(start_portfolio)
        
        trade = policy.values_in_time(t, start_portfolio, None, None, None)
        start_portfolio += trade
        
        if t in targets.index:
            assert np.all(start_portfolio == targets.loc[t])

    assert np.all(trade == 0.)


def test_sell_all(returns):

    start_portfolio = pd.Series(
        np.random.randn(
            returns.shape[1]),
        returns.columns)
    policy = SellAll()
    t = pd.Timestamp('2022-01-01')
    trade = policy.values_in_time(t, start_portfolio, None, None, None)
    allcash = np.zeros(len(start_portfolio))
    allcash[-1] = 1
    assert isinstance(trade, pd.Series)
    assert np.allclose(allcash, start_portfolio + trade)


def test_fixed_trade(returns):
    fixed_trades = pd.DataFrame(
        np.random.randn(
            len(returns),
            returns.shape[1]),
        index=returns.index,
        columns=returns.columns)

    policy = FixedTrades(fixed_trades)
    t = returns.index[123]
    trade = policy.values_in_time(t, pd.Series(
        0., returns.columns), None, None, None)
    assert np.all(trade == fixed_trades.loc[t])

    t = pd.Timestamp('1900-01-01')
    trade = policy.values_in_time(t, trade, None, None, None)
    assert np.all(trade == 0.)


def test_fixed_weights(returns):
    fixed_weights = pd.DataFrame(
        np.random.randn(
            len(returns),
            returns.shape[1]),
        index=returns.index,
        columns=returns.columns)

    policy = FixedWeights(fixed_weights)
    t = returns.index[123]
    trade = policy.values_in_time(t, pd.Series(
        0., returns.columns), None, None, None)
    assert np.all(trade == fixed_weights.loc[t])

    t = returns.index[111]
    trade = policy.values_in_time(t, fixed_weights.iloc[110], None, None, None)
    assert np.allclose(trade + fixed_weights.iloc[110], fixed_weights.loc[t])

    t = pd.Timestamp('1900-01-01')
    trade = policy.values_in_time(t, trade, None, None, None)
    assert np.all(trade == 0.)


def test_periodic_rebalance(returns):

    target = pd.Series(
        np.random.uniform(
            size=returns.shape[1]),
        returns.columns)
    target /= sum(target)
    rebalancing_times = pd.date_range(
        start=returns.index[0],
        end=returns.index[-1],
        freq='7d',
    )

    policy = PeriodicRebalance(target, rebalancing_times=rebalancing_times)
    init = pd.Series(np.random.randn(returns.shape[1]), returns.columns)

    trade = policy.values_in_time(rebalancing_times[0], init, None, None, None)
    assert np.allclose(trade + init, target)

    trade = policy.values_in_time(
        rebalancing_times[0] +
        pd.Timedelta('1d'),
        init,
        None,
        None,
        None)
    assert np.allclose(trade, 0)


def test_proportional_rebalance(returns):

    target = pd.Series(
        np.random.uniform(
            size=returns.shape[1]),
        returns.columns)
    target /= sum(target)
    target_matching_times = returns.index[::3]

    policy = ProportionalRebalance(
        target, target_matching_times=target_matching_times)
    policy.pre_evaluation(returns, None, None, None)

    init = pd.Series(np.random.randn(returns.shape[1]), returns.columns)

    trade = policy.values_in_time(returns.index[1], init, None, None, None)
    init += trade
    trade2 = policy.values_in_time(returns.index[2], init, None, None, None)
    assert np.allclose(trade, trade2)
    assert np.allclose(trade + trade2 + init, target)


def test_adaptive_rebalance(returns):
    np.random.seed(0)
    target = pd.Series(
        np.random.uniform(
            size=returns.shape[1]),
        returns.columns)
    target /= sum(target)
    target = pd.DataFrame({ind: target for ind in returns.index}).T

    init = pd.Series(np.random.uniform(size=returns.shape[1]), returns.columns)
    init /= sum(init)

    for tracking_error in [0.01, .02, .05, .1]:
        policy = AdaptiveRebalance(target, tracking_error=tracking_error)
        trade = policy.values_in_time(returns.index[1], init, None, None, None)
        assert np.allclose(init + trade, target.iloc[0])

    for tracking_error in [.2, .5]:
        policy = AdaptiveRebalance(target, tracking_error=tracking_error)
        trade = policy.values_in_time(returns.index[1], init, None, None, None)
        assert np.allclose(trade, 0.)


def test_single_period_optimization(returns, volumes):

    N = returns.shape[1]
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    policy = SinglePeriodOptimization(
        return_forecast
        - 2 * risk_forecast
        - TcostModel(half_spread=5 * 1E-4)  # , power=2)
        ,
        constraints=[LongOnly(), LeverageLimit(1)],
        # verbose=True,
        solver='ECOS')

    policy.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[50],
                          end_time=returns.index[-1])

    curw = np.zeros(N)
    curw[-1] = 1.

    result = policy.values_in_time(
        t=returns.index[51],
        current_weights=pd.Series(
            curw,
            returns.columns),
        current_portfolio_value=1000,
        past_returns=None,
        past_volumes=None)

    cvxportfolio_result = pd.Series(result, returns.columns)

    print(cvxportfolio_result)

    # REPLICATE WITH CVXPY
    w = cvx.Variable(N)
    cvx.Problem(cvx.Maximize(w.T @ return_forecast.r_hat.value -
                             2 * cvx.quad_form(w[:-1], risk_forecast.Sigma.value) -
                             5 * 1E-4 * cvx.sum(cvx.abs(w - curw)[:-1])
                             ),
                [w >= 0, w <= 1, sum(w) == 1]
                ).solve(solver='ECOS')

    cvxpy_result = pd.Series(w.value - curw, returns.columns)

    print(cvxpy_result)

    assert np.allclose(cvxportfolio_result - cvxpy_result, 0., atol=1e-7)


def test_single_period_optimization_solve_twice(returns, volumes):

    N = returns.shape[1]
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    policy = SinglePeriodOptimization(
        return_forecast
        - 2 * risk_forecast
        - TcostModel(half_spread=5 * 1E-4)  # , power=2)
        ,
        constraints=[LongOnly(), LeverageLimit(1)],
        # verbose=True,
        solver='ECOS')

    policy.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[50],
                          end_time=returns.index[-1])

    curw = np.zeros(N)
    curw[-1] = 1.

    result = policy.values_in_time(
        t=returns.index[51],
        current_weights=pd.Series(
            curw,
            returns.columns),
        current_portfolio_value=1000,
        past_returns=None,
        past_volumes=None)

    assert not np.allclose(result, 0.)

    cvxportfolio_result = pd.Series(result, returns.columns)

    curw += result

    result2 = policy.values_in_time(
        t=returns.index[51],
        current_weights=pd.Series(
            curw,
            returns.columns),
        current_portfolio_value=1000,
        past_returns=None,
        past_volumes=None)

    assert np.allclose(result2, 0., atol=1e-7)


def test_single_period_optimization_infeasible(returns, volumes):

    N = returns.shape[1]
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    policy = SinglePeriodOptimization(
        return_forecast
        - 2 * risk_forecast
        - TcostModel(half_spread=5 * 1E-4)  # , power=2)
        ,
        constraints=[LongOnly(), LeverageLimit(1), MaxWeights(-1)],
        # verbose=True,
        solver='ECOS')

    policy.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[50],
                          end_time=returns.index[-1])

    curw = np.zeros(N)
    curw[-1] = 1.

    with pytest.raises(PortfolioOptimizationError):
        result = policy.values_in_time(
            t=returns.index[51],
            current_weights=pd.Series(
                curw,
                returns.columns),
            current_portfolio_value=1000,
            past_returns=None,
            past_volumes=None)


def test_single_period_optimization_unbounded(returns, volumes):

    N = returns.shape[1]
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    policy = SinglePeriodOptimization(
        return_forecast        # - 2 * risk_forecast
        # - TcostModel(half_spread=5*1E-4)#, power=2)
        ,
        constraints=[LongOnly()],
        # verbose=True,
        solver='ECOS')

    policy.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[50],
                          end_time=returns.index[-1])

    curw = np.zeros(N)
    curw[-1] = 1.

    with pytest.raises(PortfolioOptimizationError) as e:
        result = policy.values_in_time(
            t=returns.index[51],
            current_weights=pd.Series(
                curw,
                returns.columns),
            current_portfolio_value=1000,
            past_returns=None,
            past_volumes=None)
            

def test_multi_period_optimization_syntax():
    with pytest.raises(SyntaxError):
        MultiPeriodOptimization([ReturnsForecast(rolling=50)], [])
    with pytest.raises(SyntaxError):
        MultiPeriodOptimization([ReturnsForecast(rolling=50)], [[],[]])
    with pytest.raises(SyntaxError):
        MultiPeriodOptimization([ReturnsForecast(rolling=50)], None)
    with pytest.raises(SyntaxError):
        MultiPeriodOptimization(ReturnsForecast(rolling=50))
    MultiPeriodOptimization(ReturnsForecast(rolling=50), planning_horizon = 1)
    
    
def test_multi_period_optimization1(returns, volumes):
    """Test that SPO and MPO1 return same"""
    N = returns.shape[1]
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    policy = MultiPeriodOptimization(
        return_forecast
        - 2 * risk_forecast
        - TcostModel(half_spread=5 * 1E-4)  # , power=2)
        ,
        constraints=[LongOnly(), LeverageLimit(1)],
        # verbose=True,
        planning_horizon=1,
        solver='ECOS')

    policy.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[51],
                          end_time=returns.index[-1])

    curw = np.zeros(N)
    curw[-1] = 1.

    result = policy.values_in_time(
        t=returns.index[51],
        current_weights=pd.Series(
            curw,
            returns.columns),
        current_portfolio_value=1000,
        past_returns=None,
        past_volumes=None)

    cvxportfolio_result = pd.Series(result, returns.columns)
    
    return_forecast = ReturnsForecast(rolling=50)
    risk_forecast = FullCovariance(rolling=50)
    
    policy1 = SinglePeriodOptimization(
        return_forecast
        - 2 * risk_forecast
        - TcostModel(half_spread=5 * 1E-4)  # , power=2)
        ,
        constraints=[LongOnly(), LeverageLimit(1)],
        # verbose=True,
        solver='ECOS')
        
    policy1.pre_evaluation(returns,
                          volumes,
                          start_time=returns.index[51],
                          end_time=returns.index[-1])

    result1 = policy1.values_in_time(
        t=returns.index[51],
        current_weights=pd.Series(
            curw,
            returns.columns),
        current_portfolio_value=1000,
        past_returns=None,
        past_volumes=None)
        
    cvxportfolio_result1 = pd.Series(result1, returns.columns)
        
    assert np.allclose(cvxportfolio_result - cvxportfolio_result1, 0., atol=1e-7)
    
def test_multi_period_optimization2(returns, volumes):
    """Test that MPO1 and MPO2 and MPO5 return same if no tcost, and diff if tcost"""
    
    results = []
    for planning_horizon in [1,2,5]:
        N = returns.shape[1]
        return_forecast = ReturnsForecast(rolling=50)
        risk_forecast = FullCovariance(rolling=50)
        policy = MultiPeriodOptimization(
            return_forecast
            - 10 * risk_forecast
            #- TcostModel(half_spread=5 * 1E-4)  # , power=2)
            ,
            constraints=[LongOnly(), LeverageLimit(1)],
            # verbose=True,
            planning_horizon=planning_horizon,
            solver='ECOS')

        policy.pre_evaluation(returns,
                              volumes,
                              start_time=returns.index[50],
                              end_time=returns.index[-1])

        curw = np.zeros(N)
        curw[-1] = 1.

        results.append(policy.values_in_time(
            t=returns.index[51],
            current_weights=pd.Series(
                curw,
                returns.columns),
            current_portfolio_value=1000,
            past_returns=None,
            past_volumes=None))
            
    assert np.allclose(results[0], results[1], atol=1e-4)
    assert np.allclose(results[1], results[2], atol=1e-4)
        
    # with tcost
    
    results = []
    for planning_horizon in [1,2,5]:
        N = returns.shape[1]
        return_forecast = ReturnsForecast(rolling=50)
        risk_forecast = FullCovariance(rolling=50)
        policy = MultiPeriodOptimization(
            return_forecast
            - 10 * risk_forecast
            - TcostModel(half_spread=25 * 1E-4)  # , power=2)
            ,
            constraints=[LongOnly(), LeverageLimit(1)],
            # verbose=True,
            planning_horizon=planning_horizon,
            solver='ECOS')

        policy.pre_evaluation(returns,
                              volumes,
                              start_time=returns.index[50],
                              end_time=returns.index[-1])

        curw = np.zeros(N)
        curw[-1] = 1.

        results.append(policy.values_in_time(
            t=returns.index[51],
            current_weights=pd.Series(
                curw,
                returns.columns),
            current_portfolio_value=1000,
            past_returns=None,
            past_volumes=None))
            
    assert not np.allclose(results[0], results[1], atol=1e-4)
    assert not np.allclose(results[1], results[2], atol=1e-4)
        
def test_multi_period_optimization3(returns, volumes):
    """Check that terminal constraint brings closer to benchmark."""
    
    np.random.seed(0)
    benchmark = np.random.uniform(size=returns.shape[1])
    benchmark /= sum(benchmark)
    benchmark = pd.Series(benchmark, returns.columns)
    
    diff_to_benchmarks = []
    for planning_horizon in [1,2,5]:
        N = returns.shape[1]
        return_forecast = ReturnsForecast(rolling=50)
        risk_forecast = FullCovariance(rolling=50)
        policy = MultiPeriodOptimization(
            return_forecast
            - 10 * risk_forecast
            - TcostModel(half_spread=5 * 1E-4)  # , power=2)
            ,
            constraints=[LongOnly(), LeverageLimit(1)],
            #verbose=True,
            terminal_constraint=benchmark,
            planning_horizon=planning_horizon,
            solver='ECOS')

        policy.pre_evaluation(returns,
                              volumes,
                              start_time=returns.index[50],
                              end_time=returns.index[-1])

        curw = np.zeros(N)
        curw[-1] = 1.

        diff_to_benchmarks.append(policy.values_in_time(
            t=returns.index[51],
            current_weights=pd.Series(
                curw,
                returns.columns),
            current_portfolio_value=1000,
            past_returns=None,
            past_volumes=None) + curw - benchmark)
                        
    assert np.isclose(np.linalg.norm(diff_to_benchmarks[0]), 0.)
    assert np.linalg.norm(diff_to_benchmarks[0]) < np.linalg.norm(diff_to_benchmarks[1])
    assert np.linalg.norm(diff_to_benchmarks[1]) < np.linalg.norm(diff_to_benchmarks[2])
        
        
def test_spo_old_vs_new(returns, volumes, sigma):
    tcost_model = TcostModel(half_spread=0.0005, nonlin_coeff=1.0, volume=volumes, sigma=sigma,
    )
    #hcost_model = HcostModel(borrow_costs=0., dividends=0.)
    emp_Sigma = np.cov(returns.iloc[:, :-1].to_numpy().T)
    risk_model = FullCovariance(emp_Sigma)
    returns_forecast = ReturnsForecast(returns.mean())
    
    pol = SinglePeriodOptOLD(
        returns_forecast, [
            10 * risk_model, 
            2 * tcost_model, 
            #hcost_model
        ], [], solver=cvx.ECOS)#, solver_opts={'verbose':True})#, solver_opts={'abstol':1E-16, 'reltol':1e-16, 'feastol':1e-16})
    
    #np.random.seed(0)
    init = pd.Series(1., index = returns.columns)
    init *= 1E6
        
    trad = pol.get_trades(init, t=returns.index[10])
    
    print(trad)
    
    # raise Exception

    # print(pol.costs[1].costs[0].second_term_multiplier.value)
    risk_model = FullCovariance(emp_Sigma)
    returns_forecast = ReturnsForecast(returns.mean())
    tcost_model = TcostModel(half_spread=0.0005, nonlin_coeff=1.0, volume=volumes, sigma=sigma,)
    hcost_model = HcostModel(borrow_costs=0., dividends=0.)
    
    pol_new = SinglePeriodOptNEW(
        returns_forecast, [
            10 * risk_model, 
            2 * tcost_model, 
            #hcost_model
        ], [], solver=cvx.ECOS)#, solver_opts={'verbose':True})#, solver_opts={'abstol':1E-16, 'reltol':1e-16, 'feastol':1e-16})
        
    trad_new = pol_new.get_trades(init, t=returns.index[10])
    
    print(trad_new)
    
    # print(pol_new.objective[0].costs[2].second_term_multiplier.value)
    
    assert np.allclose(trad/1E6, trad_new/1E6, atol=1e-4)
    
    ### CVXPY
    w = cvx.Variable(returns.shape[1])
    z = cvx.Variable(returns.shape[1])
    w0 = init / sum(init)
    mu = returns.mean()
    Sigma = emp_Sigma
    V = volumes.iloc[10]
    val = sum(init)
    s = sigma.iloc[10]
    objective = cvx.Maximize(w.T @ mu 
        - 10 * cvx.quad_form(w[:-1], Sigma) 
            - 2 * 0.0005 * cvx.norm1(z[:-1])
            - 2 * (cvx.abs(z[:-1])**1.5).T @ (s * np.sqrt(val / V))
            )
    cvx.Problem(objective, [cvx.sum(z) == 0, z + w0 == w]).solve(solver='ECOS')#, abstol=1E-16, reltol=1e-16, feastol=1e-16)
    
    trad_cvxpy = pd.Series(val * w.value - init, returns.columns)
    assert np.allclose((trad_cvxpy - trad_new)/sum(init), 0., atol=1E-3)        
    
    
    
        