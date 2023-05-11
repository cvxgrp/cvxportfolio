# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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

"""Unit tests for the constraints objects."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cvx
import cvxportfolio as cp


class TestConstraints(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        cls.sigma = pd.read_csv(Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
        cls.volumes = pd.read_csv(Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
        cls.w_plus = cvx.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cvx.Variable(cls.returns.shape[1])
        cls.z = cvx.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]
        
    def build_constraint(self, constraint, t=None):
        """Initialize constraint, build expression, and point it to given time."""
        constraint.pre_evaluation(self.returns.columns, self.returns.index)
        cvxpy_expression = constraint.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
        constraint.values_in_time(t=pd.Timestamp("2020-01-01") if t is None else t)
        return cvxpy_expression
        
    def test_long_only(self):
        model = cp.LongOnly()
        cons = self.build_constraint(model)
        self.w_plus.value = np.ones(self.N)
        self.assertTrue(cons.value())
        self.w_plus.value = -np.ones(self.N)
        self.assertFalse(cons.value())
        
    
    def test_long_cash(self):
        model = cp.LongCash()
        cons = self.build_constraint(model)
        self.w_plus.value = np.ones( self.N)
        self.assertTrue(cons.value())
        tmp = np.ones( self.N)
        tmp[-1] = -1
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        
    def test_dollar_neutral(self):
        model = cp.DollarNeutral()
        cons = self.build_constraint(model)
        tmpvalue = np.zeros( self.N)
        tmpvalue[-1] = 1 - sum(tmpvalue[:-1])
        self.w_plus.value = tmpvalue
        self.assertTrue(cons.value())
        tmpvalue = np.ones(self.N)
        tmpvalue[-1] = 1 - sum(tmpvalue[:-1])
        self.w_plus.value = tmpvalue
        self.assertFalse(cons.value())
        
    def test_leverage_limit(self):
        model = cp.LeverageLimit(2)
        cons = self.build_constraint(model)
        self.w_plus.value = np.ones(self.N) / self.N
        self.assertTrue(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        model = cp.LeverageLimit(7)
        cons = self.build_constraint(model)
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
    
    def test_leverage_limit_in_time(self):
        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = 7
        model = cp.LeverageLimit(limits)
        cons = self.build_constraint(model, t=self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time(t=self.returns.index[2])
        self.assertFalse(cons.value())
        
    def test_max_weights(self):
        model = cp.MaxWeights(2)
        cons = self.build_constraint(model)
        self.w_plus.value = np.ones(self.N) / self.N
        self.assertTrue(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())

        model = cp.MaxWeights(7)
        cons = self.build_constraint(model)

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = 7

        model = cp.MaxWeights(limits)
        cons = self.build_constraint(model, t=self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time(t=self.returns.index[2])
        self.assertFalse(cons.value())
        
    def test_min_weights(self):
        model = cp.MinWeights(2)
        cons = self.build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        model = cp.MinWeights(-3)
        cons = self.build_constraint(model, self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        limits = pd.Series(index=self.returns.index, data=2)
        limits.iloc[1] = -3
        model = cp.MinWeights(limits)
        cons = self.build_constraint(model, t=self.returns.index[1])
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[-1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        model.values_in_time(t=self.returns.index[2])
        self.assertFalse(cons.value())

    def test_factor_max_limit(self):
        
        model = cp.FactorMaxLimit(np.ones((self.N - 1, 2)), np.array([0.5, 1]))
        cons = self.build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertFalse(cons.value())
        

        model = cp.FactorMaxLimit(np.ones((self.N - 1, 2)), np.array([4, 4]))
        cons = self.build_constraint(model,self.returns.index[1])

        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_factor_min_limit(self):
        
        model = cp.FactorMinLimit(np.ones((self.N - 1, 2)), np.array([0.5, 1]))
        cons = self.build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        
        self.w_plus.value = tmp
        self.assertTrue(cons.value())
        
        model = cp.FactorMinLimit(np.ones((self.N - 1, 2)), np.array([4, 4]))
        cons = self.build_constraint(model, self.returns.index[1])
        # cons = model.weight_expr(t, self.w_plus, None, None)[0]
        tmp = np.zeros(self.N)
        tmp[0] = 4
        # tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

    def test_fixed_alpha(self):
        model = cp.FixedFactorLoading(np.ones((self.N - 1, 1)), 1)
        cons = self.build_constraint(model, self.returns.index[1])

        self.w_plus.value = np.ones(self.N) / self.N
        self.assertFalse(cons.value())
        tmp = np.zeros(self.N)
        tmp[0] = 4
        tmp[1] = -3
        self.w_plus.value = tmp
        self.assertTrue(cons.value())

        
    def test_participation_rate(self):
        """Test trading constraints."""

        t = self.returns.index[1]

        # avg daily value limits.
        value = 1e6
        model = cp.ParticipationRateLimit(self.volumes, max_fraction_of_volumes=0.1)
        model.pre_evaluation(self.returns.columns, self.returns.index)
        cons = model.compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
        model.values_in_time(t=t, current_portfolio_value=value)
        print(model.portfolio_value.value)
        # cons = model.weight_expr(t, None, z, value)[0]
        tmp = np.zeros(self.N)
        tmp[:-1] = self.volumes.loc[t].values / value * 0.05
        self.z.value = tmp
        self.assertTrue(cons.value())
        self.z.value = -100 * self.z.value  # -100*np.ones(n)
        self.assertFalse(cons.value())

if __name__ == '__main__':
    unittest.main()