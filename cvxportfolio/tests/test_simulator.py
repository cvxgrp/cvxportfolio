"""
Copyright (C) Enzo Busseti 2016-2019 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Code written before September 2016 is copyrighted to 
Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.,
and is licensed under the Apache License, Version 2.0 (the "License");
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
import copy

import pandas as pd

from cvxportfolio import TcostModel, HcostModel
from .base_test import BaseTest
from cvxportfolio import MarketSimulator, SimulationResult

DIR = os.path.dirname(__file__) + os.path.sep


class TestSimulator(BaseTest):

    def setUp(self):
        self.sigma = pd.read_csv(DIR + 'sigmas.csv',
                                 index_col=0, parse_dates=[0])
        self.returns = pd.read_csv(DIR + 'returns.csv',
                                   index_col=0, parse_dates=[0])
        self.volume = pd.read_csv(DIR + 'volumes.csv',
                                  index_col=0, parse_dates=[0])
        self.a, self.b, self.s = 0.0005, 1., 0.
        self.portfolio = pd.Series(index=self.returns.columns, data=1E6)
        self.tcost_term = TcostModel(self.a, self.b, self.sigma, self.volume)
        self.hcost_term = HcostModel(self.s)
        self.Simulator = MarketSimulator(self.returns,
                                         costs=[self.tcost_term,
                                                self.hcost_term],
                                         market_volumes=self.volume)

    def test_propag(self):
        """Test propagation of portfolio."""
        t = self.returns.index[1]
        h = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=h, policy=None,
                                   cash_key='cash', simulator=self.Simulator)
        u = pd.Series(index=self.portfolio.index, data=1E4)
        h_next, u = self.Simulator.propagate(h, u=u, t=t)
        results.log_simulation(t=t, u=u, h_next=h_next,
                               risk_free_return=0., exec_time=0)
        self.assertAlmostEquals(
            results.simulator_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(
            results.simulator_HcostModel.sum().sum(), 0., 3)
        self.assertAlmostEqual(sum(h_next), 28906767.251, 3)

    def test_propag_list(self):
        """Test propagation of portfolio, list of trades."""
        t = self.returns.index[1]
        h = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=h, policy=None,
                                   cash_key='cash', simulator=self.Simulator)
        u = pd.Series(index=self.portfolio.index, data=[1E4] * 29)
        h_next, u = self.Simulator.propagate(h, u, t=t)
        results.log_simulation(t=t, u=u, h_next=h_next,
                               risk_free_return=0., exec_time=0)
        self.assertAlmostEquals(
            results.simulator_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(
            results.simulator_HcostModel.sum().sum(), 0., 3)
        self.assertAlmostEqual(sum(h_next), 28906767.251, 3)

    def test_propag_neg(self):
        """Test propagation of portfolio, negative trades."""
        t = self.returns.index[1]
        h = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=h, policy=None,
                                   cash_key='cash', simulator=self.Simulator)
        u = pd.Series(index=self.portfolio.index, data=[-1E4] * 29)
        h_next, u = self.Simulator.propagate(h, u, t=t)
        results.log_simulation(t=t, u=u, h_next=h_next,
                               risk_free_return=0., exec_time=0)
        self.assertAlmostEquals(
            results.simulator_TcostModel.sum().sum(), 157.604, 3)
        self.assertAlmostEquals(
            results.simulator_HcostModel.sum().sum(), 0., 3)
        self.assertAlmostEqual(sum(h_next), 28908611.931, 3)

    def test_hcost_pos(self):
        """Test hcost function, positive positions."""
        self.hcost_term.borrow_costs += 0
        t = self.returns.index[1]
        h = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=h, policy=None,
                                   cash_key='cash', simulator=self.Simulator)
        u = pd.Series(index=self.portfolio.index, data=1E4)
        h_next, u = self.Simulator.propagate(h, u, t=t)
        results.log_simulation(t=t, u=u, h_next=h_next,
                               risk_free_return=0., exec_time=0)
        self.assertAlmostEquals(results.simulator_HcostModel.sum().sum(), 0.)

    def test_hcost_neg(self):
        """Test hcost function, negative positions."""
        self.hcost_term.borrow_costs += .0001
        t = self.returns.index[1]
        h = copy.copy(self.portfolio)
        results = SimulationResult(initial_portfolio=h, policy=None,
                                   cash_key='cash', simulator=self.Simulator)
        u = pd.Series(index=self.portfolio.index, data=-2E6)
        h_next, u = self.Simulator.propagate(h, u, t=t)
        results.log_simulation(t=t, u=u, h_next=h_next,
                               risk_free_return=0.,
                               exec_time=0)
        self.assertAlmostEquals(
            results.simulator_HcostModel.sum().sum(), 2800.0)
