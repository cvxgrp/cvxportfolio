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

"""Unit tests for the data and parameter estimator objects."""

import unittest
from pathlib import Path


import cvxpy as cp
import numpy as np
import pandas as pd


import cvxportfolio as cvx
from cvxportfolio.hyperparameters import *


class TestHyperparameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the data and initialize cvxpy vars."""
        # cls.sigma = pd.read_csv(Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(
            Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
        # cls.volumes = pd.read_csv(Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
        cls.w_plus = cp.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
        cls.z = cp.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]

    def test_basic_HP(self):

        gamma = GammaRisk(current_value=1)

        self.assertTrue((-gamma).current_value == -1)
        self.assertTrue((gamma * .5).current_value == .5)
        self.assertTrue((.5 * gamma).current_value == .5)
        self.assertTrue((gamma).current_value == 1)

        cvx.SinglePeriodOptimization(GammaRisk() * cvx.FullCovariance())

        with self.assertRaises(SyntaxError):
            cvx.SinglePeriodOptimization(GammaRisk * cvx.FullCovariance())

        cvx.SinglePeriodOptimization(-GammaRisk() * cvx.FullCovariance())

    def test_HP_algebra(self):
        grisk = GammaRisk(current_value=1)
        gtrade = GammaRisk(current_value=.5)

        self.assertTrue((grisk + gtrade).current_value == 1.5)
        self.assertTrue((grisk * gtrade).current_value == .5)
        self.assertTrue((2 * grisk * gtrade).current_value == 1)
        self.assertTrue((2 * grisk * gtrade + grisk).current_value == 2)
        self.assertTrue((1.9 * grisk * gtrade + grisk).current_value == 1.95)
        self.assertTrue((grisk + 2*gtrade).current_value == 2)
        self.assertTrue((grisk/2 + 2*gtrade).current_value == 1.5)
        self.assertTrue((grisk/2 + 2 * (gtrade + gtrade/2)).current_value == 2)

    def test_collect_HPs(self):
        """Collect hyperparameters."""

        pol = cvx.SinglePeriodOptimization(GammaRisk() * cvx.FullCovariance())

        res = pol._collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 1)

        pol = cvx.SinglePeriodOptimization(-GammaRisk() * cvx.FullCovariance()
                                           - GammaTrade() * cvx.TransactionCost())

        res = pol._collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 2)

        pol = cvx.SinglePeriodOptimization(-(GammaRisk() + .5 * GammaRisk())
                                           * cvx.FullCovariance() - GammaTrade() * cvx.TransactionCost())

        res = pol._collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 3)


if __name__ == '__main__':
    unittest.main()
