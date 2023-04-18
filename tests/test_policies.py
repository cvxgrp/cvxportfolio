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


def test_hold():
    hold = Hold()
    w = pd.Series(0.5, ["AAPL", "CASH"])
    assert np.all(
        hold.values_in_time(None, w, "extra_args", hello="ciao").values == np.zeros(2)
    )


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
    z1 = rls.values_in_time(index[0], w,  None, None, None)
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
    
    targets = pd.DataFrame({returns.index[3]: pd.Series(1., returns.columns),
                            returns.index[15]: pd.Series(-1., returns.columns)
                        }).T
    policy = ProportionalTradeToTargets(targets)
    
    policy.pre_evaluation(returns, None, None, None)
    start_portfolio = pd.Series(np.random.randn(returns.shape[1]), returns.columns)
    for t in returns.index[:17]:
        print(t)
        print(start_portfolio)
        if t in targets.index:
            assert np.all(start_portfolio == targets.loc[t])
        trade = policy.values_in_time(t, start_portfolio, None, None, None)
        start_portfolio += trade

    assert np.all(trade == 0.)
    
    
def test_sell_all(returns):
    
    start_portfolio = pd.Series(np.random.randn(returns.shape[1]), returns.columns)
    policy  = SellAll()
    t = pd.Timestamp('2022-01-01')
    trade = policy.values_in_time(t, start_portfolio, None, None, None)
    allcash = np.zeros(len(start_portfolio))
    allcash[-1] = 1
    assert isinstance(trade, pd.Series)
    assert np.allclose( allcash, start_portfolio + trade)
    
    
def test_fixed_trade(returns):
    fixed_trades = pd.DataFrame(np.random.randn(len(returns), returns.shape[1]), index=returns.index, columns=returns.columns)
    
    policy = FixedTrades(fixed_trades)
    t = returns.index[123]
    trade = policy.values_in_time(t, pd.Series(0., returns.columns), None, None, None)
    assert np.all(trade == fixed_trades.loc[t])
    
    t = pd.Timestamp('1900-01-01')
    trade = policy.values_in_time(t, trade, None, None, None)
    assert np.all(trade == 0.) 
    
    
def test_fixed_weights(returns):
    fixed_weights = pd.DataFrame(np.random.randn(len(returns), returns.shape[1]), index=returns.index, columns=returns.columns)
    
    policy = FixedWeights(fixed_weights)
    t = returns.index[123]
    trade = policy.values_in_time(t, pd.Series(0., returns.columns), None, None, None)
    assert np.all(trade == fixed_weights.loc[t])
    
    t = returns.index[111]
    trade = policy.values_in_time(t, fixed_weights.iloc[110], None, None, None)
    assert np.allclose(trade + fixed_weights.iloc[110], fixed_weights.loc[t])
    
    t = pd.Timestamp('1900-01-01')
    trade = policy.values_in_time(t, trade, None, None, None)
    assert np.all(trade == 0.) 
    
def test_periodic_rebalance(returns):
    
    target = pd.Series(np.random.uniform(size=returns.shape[1]), returns.columns)
    target /= sum(target)
    rebalancing_times=pd.date_range(
            start = returns.index[0], 
            end = returns.index[-1],
            freq = '7d',
            )
    
    policy = PeriodicRebalance(target, rebalancing_times=rebalancing_times)
    init = pd.Series(np.random.randn(returns.shape[1]), returns.columns)
    
    trade = policy.values_in_time(rebalancing_times[0], init, None, None, None)
    assert np.allclose(trade + init, target)
    
    trade = policy.values_in_time(rebalancing_times[0] + pd.Timedelta('1d'), init, None, None, None)
    assert np.allclose(trade, 0)

def test_proportional_rebalance(returns):
    
    target = pd.Series(np.random.uniform(size=returns.shape[1]), returns.columns)
    target /= sum(target)
    target_matching_times=returns.index[::3]
    
    policy = ProportionalRebalance(target, target_matching_times=target_matching_times)
    policy.pre_evaluation( returns, None, None, None)

    init = pd.Series(np.random.randn(returns.shape[1]), returns.columns)
    
    trade = policy.values_in_time(returns.index[1], init, None, None, None)
    init += trade
    trade2 = policy.values_in_time(returns.index[2], init, None, None, None)
    assert np.allclose(trade, trade2)
    assert np.allclose(trade2 + init, target)
    
def test_adaptive_rebalance(returns):
    np.random.seed(0)
    target = pd.Series(np.random.uniform(size=returns.shape[1]), returns.columns)
    target /= sum(target)
    target = pd.DataFrame({ind:target for ind in returns.index}).T
    
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
    
        
    

    
    