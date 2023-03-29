"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import cvxpy as cvx
import numpy as np
import pandas as pd

import pytest


from cvxportfolio.constraints import (LongOnly, LeverageLimit, LongCash, DollarNeutral,
                           MaxTrade, MaxWeights, MinWeights, FactorMinLimit,
                           FactorMaxLimit, FixedAlpha)
from cvxportfolio.costs import HcostModel, TcostModel
from cvxportfolio.returns import ReturnsForecast, MultipleReturnsForecasts


@pytest.fixture()
def sigma(resource_dir):
    return pd.read_csv(resource_dir / "sigmas.csv", index_col=0, parse_dates=[0])

@pytest.fixture()
def returns(resource_dir):
    return pd.read_csv(resource_dir / "returns.csv", index_col=0, parse_dates=[0])

@pytest.fixture()
def volumes(resource_dir):
    return pd.read_csv(resource_dir / "volumes.csv", index_col=0, parse_dates=[0])


#class TestModels(BaseTest):

    #def setUp(self):
        #self.sigma = pd.read_csv(DIR+'sigmas.csv',
        #                         index_col=0, parse_dates=[0])
        #self.returns = pd.read_csv(DIR+'returns.csv',
        #                           index_col=0, parse_dates=[0])
        #self.volume = pd.read_csv(DIR+'volumes.csv',
        #                          index_col=0, parse_dates=[0])
        #self.a, self.b, self.s = 0.0005, 1., 0.
        #self.universe = self.returns.columns
        #self.times = self.returns.index

def test_alpha(returns):
    """Test alpha models.
    """

    universe = returns.columns
    times = returns.index
    
    # Alpha source
    w = cvx.Variable(len(universe))
    source = ReturnsForecast(returns)
    t = times[1]
    alpha = source.weight_expr(t, w)
    w.value = np.ones(len(universe))
    assert alpha.value == pytest.approx(returns.loc[t].sum())

    # with delta
    source = ReturnsForecast(returns, returns/10)
    alpha = source.weight_expr(t, w)
    tmp = np.ones(len(universe))
    tmp[0] = -1
    w.value = tmp
    value = returns.loc[t].sum() - 2*returns.loc[t].values[0]
    value -= returns.loc[t].sum()/10

    assert alpha.value == pytest.approx(value)

    # alpha stream
    source1 = ReturnsForecast(returns)
    source2 = ReturnsForecast(-returns)
    stream = MultipleReturnsForecasts([source1, source2], [1, 1])
    alpha = stream.weight_expr(t, w)
    assert alpha.value ==  0

    stream = MultipleReturnsForecasts([source1, source2], [-1, 1])
    alpha = stream.weight_expr(t, w)
    value = returns.loc[t].sum()
    w.value = np.ones(len(universe))
    assert alpha.value == -2*value

    # with exp decay
    w = cvx.Variable(len(universe))
    source = ReturnsForecast(returns, gamma_decay=2)
    t = times[1]
    tau = times[3]
    diff = (tau - t).days
    w.value = np.ones(len(universe))
    alpha_t = source.weight_expr(t, w)
    alpha_tau = source.weight_expr_ahead(t, tau, w)
    decay = diff**(-2)
    assert alpha_tau.value == pytest.approx(decay*alpha_t.value)
