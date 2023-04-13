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

from cvxportfolio.constraints import (
    DollarNeutral,
    FactorMaxLimit,
    FactorMinLimit,
    FixedFactorLoading,
    LeverageLimit,
    LongCash,
    LongOnly,
    ParticipationRateLimit,
    MaxWeights,
    MinWeights,
)

def build_cons(model, wplus, t=None):
    model.pre_evaluation(None, None, pd.Timestamp('2022-01-01') if t is None else t, None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time(pd.Timestamp('2022-01-01') if t is None else t)
    return cons
    


def test_hold_constrs(returns):
    """Test holding constraints."""
    n = len(returns.columns)
    wplus = cvx.Variable(n)
    t = returns.index[1]

    # long only
    model = LongOnly()
    cons = build_cons(model, wplus)
    wplus.value = np.ones(n)
    assert cons.value()
    wplus.value = -np.ones(n)
    assert not cons.value()
    

    # long cash
    model = LongCash()
    cons = build_cons(model, wplus)
    wplus.value = np.ones(n)
    assert cons.value()
    tmp = np.ones(n)
    tmp[-1] = -1
    wplus.value = tmp
    assert not cons.value()
    

    # dollar neutral
    model = DollarNeutral()
    cons = build_cons(model, wplus)
    wplus.value = np.zeros(n)
    assert cons.value()
    wplus.value = np.ones(n)
    assert not cons.value()
    

    # leverage limit
    model = LeverageLimit(2)
    cons = build_cons(model, wplus, t=None)
    wplus.value = np.ones(n) / n
    assert cons.value()
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    model = LeverageLimit(7)
    cons = build_cons(model, wplus, t=None)
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    
    # leverage limit in time!

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = 7
    model = LeverageLimit(limits)
    cons = build_cons(model, wplus, t=returns.index[1])
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    model.values_in_time(returns.index[2])
    assert not cons.value()
    
    

    # Max weights
    model = MaxWeights(2)
    cons = build_cons(model, wplus)
    
    wplus.value = np.ones(n) / n
    assert cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    
    
    model = MaxWeights(7)
    cons = build_cons(model, wplus)
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    
    

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = 7
    
    model = MaxWeights(limits)
    cons = build_cons(model, wplus, returns.index[1])
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    model.values_in_time(returns.index[2])
    assert not cons.value()
    
    

    # Min weights
    model = MinWeights(2)
    cons = build_cons(model, wplus, returns.index[1])
    
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    model = MinWeights(-3)
    cons = build_cons(model, wplus, returns.index[1])

    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = -3
    model = MinWeights(limits)
    cons = build_cons(model, wplus, returns.index[1])
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    model.values_in_time(returns.index[2])
    assert not cons.value()


    # Factor Max Limit
    model = FactorMaxLimit(np.ones((n - 1, 2)), np.array([0.5, 1]))
    cons = build_cons(model, wplus, returns.index[1])
    
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert not cons.value()
    
    model = FactorMaxLimit(np.ones((n - 1, 2)), np.array([4, 4]))
    cons = build_cons(model, wplus, returns.index[1])
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()
    

    # Factor Min Limit
    model = FactorMinLimit(np.ones((n - 1, 2)), np.array([0.5, 1]))
    cons = build_cons(model, wplus, returns.index[1])
    #cons = model.weight_expr(t, wplus, None, None)[0]
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()
    model = FactorMinLimit(np.ones((n - 1, 2)), np.array([4, 4]))
    cons = build_cons(model, wplus, returns.index[1])
    #cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    # tmp[1] = -3
    wplus.value = tmp
    assert cons.value()
    

    # Fixed Alpha
    model = FixedFactorLoading(np.ones((n - 1, 1)), 1)
    cons = build_cons(model, wplus, returns.index[1])
    
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()    


def test_trade_constr(returns, volumes):
    """Test trading constraints."""
    n = len(returns.columns)
    z = cvx.Variable(n)
    t = returns.index[1]
    

    # avg daily value limits.
    value = 1e6
    model = ParticipationRateLimit(volumes, max_fraction_of_volumes=0.1)
    model.pre_evaluation(None, None, returns.index[0], None)
    cons = model.compile_to_cvxpy(None, z, value)[0]
    model.values_in_time(t)
    #cons = model.weight_expr(t, None, z, value)[0]
    tmp = np.zeros(n)
    tmp[:-1] = volumes.loc[t].values / value * 0.05
    z.value = tmp
    assert cons.value()
    z.value = -100 * z.value  # -100*np.ones(n)
    assert not cons.value()