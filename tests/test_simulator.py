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

import pandas as pd
import pytest

from cvxportfolio import TcostModel, HcostModel
from cvxportfolio import MarketSimulator, SimulationResult


@pytest.fixture()
def portfolio(returns):
    return pd.Series(index=returns.columns, data=1E6)


@pytest.fixture()
def hcost_term():
    return HcostModel(0.0)

@pytest.fixture()
def tcost_term(sigma, volumes):
    return TcostModel(0.0005, 1.0, sigma, volumes)

@pytest.fixture()
def simulator(returns, hcost_term, tcost_term, volumes):
    return MarketSimulator(returns, costs=[tcost_term, hcost_term], market_volumes=volumes)

    #def setUp(self):
    #    self.a, self.b, self.s = 0.0005, 1., 0.
    #    self.portfolio = pd.Series(index=self.returns.columns, data=1E6)
    #    self.tcost_term = TcostModel(self.a, self.b, self.sigma, self.volume)
    #    self.hcost_term = HcostModel(self.s)
    #    self.Simulator = MarketSimulator(self.returns,
    #                                     costs=[self.tcost_term,
    #                                            self.hcost_term],
    #                                     market_volumes=self.volume)

def test_propag(returns, portfolio, simulator):
    """Test propagation of portfolio."""
    t = returns.index[1]
    results = SimulationResult(initial_portfolio=portfolio, policy=None,
                               cash_key='cash', simulator=simulator)
    u = pd.Series(index=portfolio.index, data=1E4)
    h_next, u = simulator.propagate(portfolio, u=u, t=t)
    results.log_simulation(t=t, u=u, h_next=h_next,
                           risk_free_return=0., exec_time=0)
    assert results.simulator_TcostModel.sum().sum() == pytest.approx(157.604, abs=1e-3)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(0.0, abs=1e-3)
    assert sum(h_next) == pytest.approx(28906767.251, abs=1e-3)

def test_propag_list(returns, portfolio, simulator):
    """Test propagation of portfolio, list of trades."""
    t = returns.index[1]
    results = SimulationResult(initial_portfolio=portfolio, policy=None,
                               cash_key='cash', simulator=simulator)
    u = pd.Series(index=portfolio.index, data=[1E4]*29)
    h_next, u = simulator.propagate(portfolio, u, t=t)
    results.log_simulation(t=t, u=u, h_next=h_next,
                           risk_free_return=0., exec_time=0)
    assert results.simulator_TcostModel.sum().sum() == pytest.approx(157.604, abs=1e-3)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(0.0, abs=1e-3)
    assert sum(h_next) == pytest.approx(28906767.251, abs=1e-3)

    
def test_propag_neg(returns, portfolio, simulator):
    """Test propagation of portfolio, negative trades."""
    t = returns.index[1]
    results = SimulationResult(initial_portfolio=portfolio, policy=None,
                               cash_key='cash', simulator=simulator)
    u = pd.Series(index=portfolio.index, data=[-1E4]*29)
    h_next, u = simulator.propagate(portfolio, u, t=t)
    results.log_simulation(t=t, u=u, h_next=h_next,
                           risk_free_return=0., exec_time=0)
    assert results.simulator_TcostModel.sum().sum() == pytest.approx(157.604, abs=1e-3)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(0.0, abs=1e-3)
    assert sum(h_next) == pytest.approx(28908611.931, abs=1e-3)


    
def test_hcost_pos(returns):
    """Test hcost function, positive positions."""
    self.hcost_term.borrow_costs += 0
    t = returns.index[1]
    h = copy.copy(self.portfolio)
    results = SimulationResult(initial_portfolio=h, policy=None,
                               cash_key='cash', simulator=self.Simulator)
    u = pd.Series(index=self.portfolio.index, data=1E4)
    h_next, u = self.Simulator.propagate(h, u, t=t)
    results.log_simulation(t=t, u=u, h_next=h_next,
                           risk_free_return=0., exec_time=0)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(0.0, abs=1e-8)
    
    #self.assertAlmostEquals(results.simulator_HcostModel.sum().sum(), 0.)

    
def test_hcost_neg(portfolio):
    """Test hcost function, negative positions."""
    self.hcost_term.borrow_costs += .0001
    t = self.returns.index[1]
    #h = copy.copy(self.portfolio)
    results = SimulationResult(initial_portfolio=portfolio, policy=None,
                               cash_key='cash', simulator=self.Simulator)
    u = pd.Series(index=portfolio.index, data=-2E6)
    h_next, u = self.Simulator.propagate(h, u, t=t)
    results.log_simulation(t=t, u=u, h_next=h_next,
                           risk_free_return=0.,
                           exec_time=0)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(2800.0)