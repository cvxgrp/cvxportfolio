"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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
import pickle
import copy

import pandas as pd

from cvx_portfolio.returns import AlphaSource, MarketReturns
from .base_test import BaseTest
from ..costs import TcostModel, HcostModel
from ..portfolio import Portfolio
from ..simulator.market import MarketSimulator
from ..simulator.result import SimulationResult

DATAFILE = os.path.dirname(__file__) + os.path.sep + 'sample_data.pickle'


class TestSimulator(BaseTest):

    def setUp(self):
        with open(DATAFILE, 'rb') as f:
            self.returns, self.sigma, self.volume, self.a, self.b, self.s = \
            pickle.load(f)
        self.portfolio = Portfolio(pd.Series(index = self.returns.columns, data=1E6))
        returns_model = MarketReturns(self.returns)
        self.tcost_term = TcostModel(self.volume, self.sigma, self.a, self.b)
        self.hcost_term = HcostModel(self.s)
        self.Simulator = MarketSimulator(returns_model, costs=[self.tcost_term, self.hcost_term])

    def test_propag(self):
        """Test propagation of portfolio."""
        t = self.returns.index[1]
        next_portf = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=next_portf, policy=None,
                                   simulator=self.Simulator)
        self.Simulator.propagate(next_portf,
                                    u=pd.Series(index=self.portfolio.h.index,
                                                data=1E4), t=t)
        self.assertAlmostEquals(results.sim_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(results.sim_HcostModel.sum(), 0., 3)
        self.assertAlmostEqual(next_portf.v, 28906767.251, 3)

    def test_propag_list(self):
        """Test propagation of portfolio, list of trades."""
        t = self.returns.index[1]
        next_portf = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=next_portf, policy=None,
                                   simulator=self.Simulator)
        self.Simulator.propagate(next_portf,
                                     pd.Series(index=self.portfolio.h.index, data=[1E4]*29),
                                     t=t)
        self.assertAlmostEquals(results.sim_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(results.sim_HcostModel.sum(), 0., 3)
        self.assertAlmostEqual(next_portf.v, 28906767.251, 3)

    def test_propag_neg(self):
        """Test propagation of portfolio, negative trades."""
        t = self.returns.index[1]
        next_portf = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=next_portf, policy=None,
                                   simulator=self.Simulator)
        self.Simulator.propagate(next_portf,
                                     pd.Series(index=self.portfolio.h.index, data=[-1E4]*29),
                                     t=t)
        self.assertAlmostEquals(results.sim_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(results.sim_HcostModel.sum(), 0., 3)
        self.assertAlmostEqual(next_portf.v, 28908611.931, 3)

    def test_hcost_pos(self):
        """Test hcost function, positive positions."""
        self.hcost_term.borrow_costs += 1
        t = self.returns.index[1]
        next_portf = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=next_portf, policy=None,
                                   simulator=self.Simulator)
        self.Simulator.propagate(next_portf,
                                     u=pd.Series(index=self.portfolio.h.index,
                                                data=1E4), t=t)

        self.assertAlmostEquals(results.sim_HcostModel.sum(), 0.)

    def test_hcost_neg(self):
        """Test hcost function, negative positions."""
        self.hcost_term.borrow_costs += .0001
        t = self.returns.index[1]
        next_portf = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=next_portf, policy=None,
                                   simulator=self.Simulator)
        self.Simulator.propagate(next_portf,
                                     u=pd.Series(index=self.portfolio.h.index,
                                                data=-2E6), t=t)

        self.assertAlmostEquals(results.sim_HcostModel.sum(), 2800.0)
