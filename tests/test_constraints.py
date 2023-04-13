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
    MaxTrade,
    MaxWeights,
    MinWeights,
)

def test_hold_constrs(returns):
    """Test holding constraints."""
    n = len(returns.columns)
    wplus = cvx.Variable(n)
    t = returns.index[1]

    # long only
    model = LongOnly()
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    wplus.value = np.ones(n)
    assert cons.value()
    wplus.value = -np.ones(n)
    assert not cons.value()
    

    # long cash
    model = LongCash()
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    wplus.value = np.ones(n)
    assert cons.value()
    tmp = np.ones(n)
    tmp[-1] = -1
    wplus.value = tmp
    assert not cons.value()
    
    

    # dollar neutral
    model = DollarNeutral()
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    wplus.value = np.zeros(n)
    assert cons.value()
    wplus.value = np.ones(n)
    assert not cons.value()
    

    # leverage limit
    model = LeverageLimit(2)
    model.pre_evaluation(None, None, pd.Timestamp('2022-01-01'), None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time(pd.Timestamp('2022-01-01'))
    wplus.value = np.ones(n) / n
    assert cons.value()
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    model = LeverageLimit(7)
    model.pre_evaluation(None, None, pd.Timestamp('2022-01-01'), None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time(pd.Timestamp('2022-01-01'))
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    
    # leverage limit in time!

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = 7
    model = LeverageLimit(limits)
    model.pre_evaluation(None, None, returns.index[0], None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time(returns.index[1])
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    model.values_in_time(returns.index[2])
    assert not cons.value()
    
    

    # Max weights
    model = MaxWeights(2)
    model.pre_evaluation(None, None, pd.Timestamp('2022-01-01'), None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time('2020-01-01')
    
    wplus.value = np.ones(n) / n
    assert cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    
    model = MaxWeights(7)
    model.pre_evaluation(None, None, pd.Timestamp('2022-01-01'), None)
    cons = model.compile_to_cvxpy(wplus, None, None)[0]
    model.values_in_time('2020-01-01')
    
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    
    raise Exception
    

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = 7
    model = MaxWeights(limits)
    cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    cons = model.weight_expr(returns.index[2], wplus, None, None)[0]
    assert not cons.value()

    # Min weights
    model = MinWeights(2)
    cons = model.weight_expr(t, wplus, None, None)[0]
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert not cons.value()
    model = MinWeights(-3)
    cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()

    limits = pd.Series(index=returns.index, data=2)
    limits.iloc[1] = -3
    model = MinWeights(limits)
    cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[-1] = -3
    wplus.value = tmp
    assert cons.value()
    cons = model.weight_expr(returns.index[2], wplus, None, None)[0]
    assert not cons.value()

    # Factor Max Limit
    model = FactorMaxLimit(np.ones((n - 1, 2)), [0.5, 1])
    cons = model.weight_expr(t, wplus, None, None)[0]
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert not cons.value()
    model = FactorMaxLimit(np.ones((n - 1, 2)), [4, 4])
    cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()

    # Factor Min Limit
    model = FactorMinLimit(np.ones((n - 1, 2)), [0.5, 1])
    cons = model.weight_expr(t, wplus, None, None)[0]
    wplus.value = np.ones(n) / n
    assert not cons.value()
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()
    model = FactorMinLimit(np.ones((n - 1, 2)), [-4, -4])
    cons = model.weight_expr(t, wplus, None, None)[0]
    tmp = np.zeros(n)
    tmp[0] = 4
    tmp[1] = -3
    wplus.value = tmp
    assert cons.value()

    # Fixed Alpha
    model = FixedAlpha(np.ones((n - 1, 1)), 1)
    cons = model.weight_expr(t, wplus, None, None)[0]
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
    model = MaxTrade(volumes, max_fraction=0.1)
    cons = model.weight_expr(t, None, z, value)[0]
    tmp = np.zeros(n)
    tmp[:-1] = volumes.loc[t].values / value * 0.05
    z.value = tmp
    assert cons.value()
    z.value = -100 * z.value  # -100*np.ones(n)
    assert not cons.value()