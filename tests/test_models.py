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

from .base_test import BaseTest
from ..constraints import (LongOnly, LeverageLimit, LongCash, DollarNeutral,
                           MaxTrade, MaxWeights, MinWeights, FactorMinLimit,
                           FactorMaxLimit, FixedAlpha)
from ..costs import HcostModel, TcostModel
from ..returns import ReturnsForecast, MultipleReturnsForecasts

DIR = os.path.dirname(__file__) + os.path.sep


class TestModels(BaseTest):

    def setUp(self):
        self.sigma = pd.read_csv(DIR+'sigmas.csv',
                                 index_col=0, parse_dates=[0])
        self.returns = pd.read_csv(DIR+'returns.csv',
                                   index_col=0, parse_dates=[0])
        self.volume = pd.read_csv(DIR+'volumes.csv',
                                  index_col=0, parse_dates=[0])
        self.a, self.b, self.s = 0.0005, 1., 0.
        self.universe = self.returns.columns
        self.times = self.returns.index

    def test_alpha(self):
        """Test alpha models.
        """
        # Alpha source
        w = cvx.Variable(len(self.universe))
        source = ReturnsForecast(self.returns)
        t = self.times[1]
        alpha = source.weight_expr(t, w)
        w.value = np.ones(len(self.universe))
        self.assertAlmostEqual(alpha.value, self.returns.loc[t].sum())
        # with delta
        source = ReturnsForecast(self.returns, self.returns/10)
        alpha = source.weight_expr(t, w)
        tmp = np.ones(len(self.universe))
        tmp[0] = -1
        w.value = tmp
        value = self.returns.loc[t].sum() - 2*self.returns.loc[t].values[0]
        value -= self.returns.loc[t].sum()/10
        self.assertAlmostEqual(alpha.value, value)

        # alpha stream
        source1 = ReturnsForecast(self.returns)
        source2 = ReturnsForecast(-self.returns)
        stream = MultipleReturnsForecasts([source1, source2], [1, 1])
        alpha = stream.weight_expr(t, w)
        self.assertEqual(alpha.value, 0)

        stream = MultipleReturnsForecasts([source1, source2], [-1, 1])
        alpha = stream.weight_expr(t, w)
        value = self.returns.loc[t].sum()
        w.value = np.ones(len(self.universe))
        self.assertEqual(alpha.value, -2*value)

        # with exp decay
        w = cvx.Variable(len(self.universe))
        source = ReturnsForecast(self.returns, gamma_decay=2)
        t = self.times[1]
        tau = self.times[3]
        diff = (tau - t).days
        w.value = np.ones(len(self.universe))
        alpha_t = source.weight_expr(t, w)
        alpha_tau = source.weight_expr_ahead(t, tau, w)
        decay = diff**(-2)
        self.assertAlmostEqual(alpha_tau.value, decay*alpha_t.value)

    def test_tcost_value_expr(self):
        """Test the value expression of the tcost.
        """
        n = len(self.universe)
        value = 1e6
        model = TcostModel(half_spread=self.a, nonlin_coeff=0.,
                           sigma=self.sigma, volume=self.volume)
        t = self.times[1]
        z = np.arange(n) - n/2
        z_var = cvx.Variable(n)
        z_var.value = z
        tcost, _ = model.weight_expr(t, None, z_var, value)
        u = pd.Series(index=self.returns.columns, data=z_var.value*value)
        value_expr = model.value_expr(t, None, u)
        self.assertAlmostEqual(tcost.value, value_expr/value)

        model = TcostModel(half_spread=0, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume, power=2)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        self.b * self.sigma.loc[t] * (value / self.volume.loc[t])
        value_expr = model.value_expr(t, None, u)
        self.assertAlmostEqual(tcost.value, value_expr/value)

        model = TcostModel(half_spread=0, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume, power=1.5)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        self.b * self.sigma.loc[t] * np.sqrt(value / self.volume.loc[t])
        value_expr = model.value_expr(t, None, u)
        self.assertAlmostEqual(tcost.value, value_expr/value)

        model = TcostModel(half_spread=self.a, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        value_expr = model.value_expr(t, None, u)
        self.assertAlmostEqual(tcost.value, value_expr/value)

        # with tau
        model = TcostModel(half_spread=self.a, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume)
        tau = self.times[2]
        tcost, _ = model.weight_expr_ahead(t, tau, None, z_var, value)
        value_expr = model.value_expr(t, None, u)
        self.assertAlmostEqual(tcost.value, value_expr/value)

    def test_tcost(self):
        """Test tcost model.
        """
        n = len(self.universe)
        value = 1e6
        model = TcostModel(half_spread=self.a, nonlin_coeff=0.,
                           sigma=self.sigma, volume=self.volume)
        t = self.times[1]
        z = np.arange(n) - n/2
        z_var = cvx.Variable(n)
        z_var.value = z
        tcost, _ = model.weight_expr(t, None, z_var, value)
        est_tcost_lin = sum(np.abs(z[:-1])*self.a)
        self.assertAlmostEqual(tcost.value, est_tcost_lin)

        model = TcostModel(half_spread=0, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume, power=2)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        coeff = self.b * self.sigma.loc[t] * \
            (value / self.volume.loc[t])
        est_tcost_nonlin = np.square(z[:-1]).dot(coeff.values)
        self.assertAlmostEqual(tcost.value, est_tcost_nonlin)

        model = TcostModel(half_spread=0, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume, power=1.5)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        coeff = self.b * self.sigma.loc[t] * \
            np.sqrt(value / self.volume.loc[t])
        est_tcost_nonlin = np.power(np.abs(z[:-1]), 1.5).dot(coeff.values)
        self.assertAlmostEqual(tcost.value, est_tcost_nonlin)

        model = TcostModel(half_spread=self.a, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume)
        tcost, _ = model.weight_expr(t, None, z_var, value)
        self.assertAlmostEqual(tcost.value, est_tcost_nonlin + est_tcost_lin)

        # with tau
        model = TcostModel(half_spread=self.a, nonlin_coeff=self.b,
                           sigma=self.sigma, volume=self.volume)
        tau = self.times[2]
        tcost, _ = model.weight_expr_ahead(t, tau, None, z_var, value)
        self.assertAlmostEqual(tcost.value, est_tcost_nonlin + est_tcost_lin)

        tau = t + 10*pd.Timedelta('1 days')
        tcost_tau, _ = model.est_period(t, t, tau, None, z_var, value)
        tcost_t, _ = model.weight_expr(t, None, z_var / 10, value)
        tcost_t *= 10
        self.assertAlmostEqual(tcost_tau.value, tcost_t.value)

    def test_hcost(self):
        """Test holding cost model.
        """
        div = self.s/2
        n = len(self.universe)
        wplus = cvx.Variable(n)
        wplus.value = np.arange(n) - n/2
        t = self.times[1]
        model = HcostModel(self.s)
        hcost, _ = model.weight_expr(t, wplus, None, None)
        bcost = sum(wplus[:-1].value*self.s)
        self.assertAlmostEqual(hcost.value, bcost)

        model = HcostModel(self.s*0, div)
        hcost, _ = model.weight_expr(t, wplus, None, None)
        divs = (np.sum(wplus[:-1].value*div))
        self.assertAlmostEqual(-hcost.value, divs)

        model = HcostModel(self.s, div)
        hcost, _ = model.weight_expr(t, wplus, None, None)
        self.assertAlmostEqual(hcost.value, bcost - divs)

    def test_hcost_value_expr(self):
        """Test the value expression of the hcost.
        """
        div = self.s/2
        n = len(self.universe)
        wplus = cvx.Variable(n)
        wplus.value = np.arange(n) - n/2
        t = self.times[1]
        model = HcostModel(self.s)
        hcost, _ = model.weight_expr(t, wplus, None, None)

        value = 1000.
        h_plus = pd.Series(index=self.returns.columns,
                           data=wplus.value*1000)
        value_expr = model.value_expr(t, h_plus, None)

        self.assertAlmostEqual(hcost.value, value_expr/value)

        model = HcostModel(self.s*0, div)
        hcost, _ = model.weight_expr(t, wplus, None, None)
        value_expr = model.value_expr(t, h_plus, None)
        self.assertAlmostEqual(-hcost.value, value_expr/value)

        model = HcostModel(self.s, div)
        hcost, _ = model.weight_expr(t, wplus, None, None)
        value_expr = model.value_expr(t, h_plus, None)
        self.assertAlmostEqual(hcost.value, value_expr/value)

    def test_hold_constrs(self):
        """Test holding constraints.
        """
        n = len(self.universe)
        wplus = cvx.Variable(n)
        t = self.times[1]

        # long only
        model = LongOnly()
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n)
        assert cons.value()
        wplus.value = -np.ones(n)
        assert not cons.value()

        # long cash
        model = LongCash()
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n)
        assert cons.value()
        tmp = np.ones(n)
        tmp[-1] = -1
        wplus.value = tmp
        assert not cons.value()

        # dollar neutral
        model = DollarNeutral()
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.zeros(n)
        assert cons.value()
        wplus.value = np.ones(n)
        assert not cons.value()

        # leverage limit
        model = LeverageLimit(2)
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n)/n
        assert cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert not cons.value()
        model = LeverageLimit(7)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()

        limits = pd.Series(index=self.times, data=2)
        limits.iloc[1] = 7
        model = LeverageLimit(limits)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()
        cons = model.weight_expr(self.times[2], wplus, None, None)
        assert not cons.value()

        # Max weights
        model = MaxWeights(2)
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n) / n
        assert cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert not cons.value()
        model = MaxWeights(7)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()

        limits = pd.Series(index=self.times, data=2)
        limits.iloc[1] = 7
        model = MaxWeights(limits)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()
        cons = model.weight_expr(self.times[2], wplus, None, None)
        assert not cons.value()

        # Min weights
        model = MinWeights(2)
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n) / n
        assert not cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert not cons.value()
        model = MinWeights(-3)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()

        limits = pd.Series(index=self.times, data=2)
        limits.iloc[1] = -3
        model = MinWeights(limits)
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[-1] = -3
        wplus.value = tmp
        assert cons.value()
        cons = model.weight_expr(self.times[2], wplus, None, None)
        assert not cons.value()

        # Factor Max Limit
        model = FactorMaxLimit(np.ones((n - 1, 2)), [.5, 1])
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n) / n
        assert not cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[1] = -3
        wplus.value = tmp
        assert not cons.value()
        model = FactorMaxLimit(np.ones((n - 1, 2)), [4, 4])
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[1] = -3
        wplus.value = tmp
        assert cons.value()

        # Factor Min Limit
        model = FactorMinLimit(np.ones((n - 1, 2)), [.5, 1])
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n) / n
        assert not cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[1] = -3
        wplus.value = tmp
        assert cons.value()
        model = FactorMinLimit(np.ones((n - 1, 2)), [-4, -4])
        cons = model.weight_expr(t, wplus, None, None)
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[1] = -3
        wplus.value = tmp
        assert cons.value()

        # Fixed Alpha
        model = FixedAlpha(np.ones((n - 1, 1)), 1)
        cons = model.weight_expr(t, wplus, None, None)
        wplus.value = np.ones(n) / n
        assert not cons.value()
        tmp = np.zeros(n)
        tmp[0] = 4
        tmp[1] = -3
        wplus.value = tmp
        assert cons.value()

    def test_trade_constr(self):
        """Test trading constraints.
        """
        n = len(self.universe)
        z = cvx.Variable(n)
        t = self.times[1]

        # avg daily value limits.
        value = 1e6
        model = MaxTrade(self.volume, max_fraction=.1)
        cons = model.weight_expr(t, None, z, value)
        tmp = np.zeros(n)
        tmp[:-1] = self.volume.loc[t].values / value * 0.05
        z.value = tmp
        assert cons.value()
        z.value = -100*z.value  # -100*np.ones(n)
        assert not cons.value()
