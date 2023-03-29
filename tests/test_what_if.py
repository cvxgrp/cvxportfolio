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
import copy

from cvxportfolio import simulator, HcostModel, TcostModel, SinglePeriodOpt
from cvxportfolio import ReturnsForecast, MultipleReturnsForecasts, FullSigma
from .base_test import BaseTest

DIR = os.path.dirname(__file__) + os.path.sep


class TestWhatIf(BaseTest):

    def setUp(self):
        self.sigma = pd.read_csv(DIR + 'sigmas.csv',
                                 index_col=0, parse_dates=[0])
        self.returns = pd.read_csv(DIR + 'returns.csv',
                                   index_col=0, parse_dates=[0])
        self.volume = pd.read_csv(DIR + 'volumes.csv',
                                  index_col=0, parse_dates=[0])
        self.a, self.b, self.s = 0.0005, 1., 0.
        self.universe = self.returns.columns
        self.times = self.returns.index

    def test_attribution(self):
        """Test attribution.
        """
        # Alpha source
        alpha_sources = [ReturnsForecast(
            self.returns, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        emp_Sigma = np.cov(self.returns.to_numpy().T)
        risk_model = FullSigma(emp_Sigma, gamma=100.)
        tcost_model = TcostModel(self.volume, self.sigma, self.a, self.b)
        hcost_model = HcostModel(self.s, self.s * 0)
        pol = SinglePeriodOpt(alpha_model, [risk_model, tcost_model,
                                            hcost_model], [],
                              solver=cvx.ECOS)

        tcost = TcostModel(self.a, self.b, self.sigma, self.volume)
        hcost = HcostModel(self.s)
        market_sim = simulator.MarketSimulator(self.returns,
                                               costs=[tcost, hcost],
                                               market_volumes=self.volume)

        p_0 = pd.Series(index=self.universe, data=1E6)
        noisy = market_sim.run_backtest(p_0, self.returns.index[1],
                                        self.returns.index[10], pol)
        # linear fit attribution
        attr = market_sim.attribute(noisy, pol,
                                    parallel=False, fit="linear")
        base_line = noisy.v - sum(p_0)
        for i in range(3):
            self.assertItemsAlmostEqual(
                attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))
        self.assertItemsAlmostEqual(attr['RMS error'], np.zeros(len(noisy.v)))

        # least-squares fit attribution
        attr = market_sim.attribute(noisy, pol,
                                    parallel=False, fit="least-squares")
        base_line = noisy.v - sum(p_0)
        for i in range(3):
            self.assertItemsAlmostEqual(
                attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))
        # Residual always 0.
        alpha_sources = [ReturnsForecast(
            self.returns * 0, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        pol = copy.copy(pol)
        pol.alpha_model = alpha_model
        attr = market_sim.attribute(noisy, pol,
                                    parallel=False, fit="least-squares")
        self.assertItemsAlmostEqual(attr['residual'], np.zeros(len(noisy.v)))

    def test_attribute_non_profit_series(self):
        """Test attributing series quantities besides profit.
        """
        # Alpha source
        alpha_sources = [ReturnsForecast(
            self.returns, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        emp_Sigma = np.cov(self.returns.to_numpy().T)
        risk_model = FullSigma(emp_Sigma, gamma=100.)
        tcost_model = TcostModel(self.a, self.b, self.sigma, self.volume)
        hcost_model = HcostModel(self.s, self.s * 0)
        pol = SinglePeriodOpt(alpha_model, [risk_model, tcost_model,
                                            hcost_model], [],
                              solver=cvx.ECOS)

        tcost = TcostModel(self.volume, self.sigma, self.a, self.b)
        hcost = HcostModel(self.s)
        market_sim = simulator.MarketSimulator(self.returns,
                                               costs=[tcost, hcost],
                                               market_volumes=self.volume)

        p_0 = pd.Series(index=self.universe, data=1E6)
        noisy = market_sim.run_backtest(p_0, self.returns.index[1],
                                        self.returns.index[10], pol)
        # Select tcosts.

        def selector(result):
            return result.leverage

        # linear fit attribution
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="linear")
        base_line = noisy.leverage
        for i in range(3):
            self.assertItemsAlmostEqual(
                attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))
        self.assertItemsAlmostEqual(attr['RMS error'], np.zeros(len(noisy.v)))

        # least-squares fit attribution
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="least-squares")
        for i in range(3):
            self.assertItemsAlmostEqual(
                attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))
        # Residual always 0.
        alpha_sources = [ReturnsForecast(
            self.returns * 0, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        pol = copy.copy(pol)
        pol.alpha_model = alpha_model
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="least-squares")
        self.assertItemsAlmostEqual(attr['residual'], np.zeros(len(noisy.v)))

    def test_attribute_non_profit_scalar(self):
        """Test attributing scalar quantities besides profit.
        """
        # Alpha source
        alpha_sources = [ReturnsForecast(
            self.returns, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        emp_Sigma = np.cov(self.returns.to_numpy().T)
        risk_model = FullSigma(emp_Sigma)
        tcost_model = TcostModel(self.a, self.b, self.sigma, self.volume)
        hcost_model = HcostModel(self.s)
        pol = SinglePeriodOpt(
            alpha_model, [100 * risk_model, tcost_model, hcost_model], [])

        market_sim = simulator.MarketSimulator(self.returns,
                                               costs=[tcost_model, hcost_model]
                                               )

        p_0 = pd.Series(index=self.universe, data=1E6)
        noisy = market_sim.run_backtest(p_0, self.returns.index[1],
                                        self.returns.index[10], pol)
        # Select tcosts.

        def selector(result):
            return pd.Series(index=[noisy.h.index[-1]],
                             data=result.volatility)

        # linear fit attribution
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="linear")
        base_line = noisy.volatility
        for i in range(3):
            self.assertAlmostEqual(
                attr[i][0] / weights[i] / sum(p_0), base_line / sum(p_0))
        self.assertItemsAlmostEqual(attr['RMS error'], np.zeros(len(noisy.v)))

        # least-squares fit attribution
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="least-squares")
        for i in range(3):
            self.assertAlmostEqual(
                attr[i][0] / weights[i] / sum(p_0), base_line / sum(p_0))
        # Residual always 0.
        alpha_sources = [ReturnsForecast(
            self.returns * 0, name=i) for i in range(3)]
        weights = np.array([0.1, 0.3, 0.6])
        alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
        pol = copy.copy(pol)
        pol.alpha_model = alpha_model
        attr = market_sim.attribute(noisy, pol, selector,
                                    parallel=False, fit="least-squares")
        self.assertItemsAlmostEqual(attr['residual'], np.zeros(len(noisy.v)))
