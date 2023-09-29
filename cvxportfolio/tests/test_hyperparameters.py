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
"""Unit tests for the hyper-parameters interface."""

import unittest

import cvxpy as cp
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.hyperparameters import GammaRisk, GammaTrade
from cvxportfolio.tests import CvxportfolioTest


class TestHyperparameters(CvxportfolioTest):
    """Test hyper-parameters interface."""

    def test_basic_hyper_parameters(self):
        """Test simple syntax and errors."""
        gamma = GammaRisk(current_value=1)

        self.assertTrue((-gamma).current_value == -1)
        self.assertTrue((gamma * .5).current_value == .5)
        self.assertTrue((.5 * gamma).current_value == .5)
        self.assertTrue((gamma).current_value == 1)

        cvx.SinglePeriodOptimization(GammaRisk() * cvx.FullCovariance())

        with self.assertRaises(SyntaxError):
            cvx.SinglePeriodOptimization(GammaRisk * cvx.FullCovariance())

        cvx.SinglePeriodOptimization(-GammaRisk() * cvx.FullCovariance())

    def test_hyper_parameters_algebra(self):
        """Test algebra of HPs objects."""
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

    def test_collect_hyper_parameters(self):
        """Test collect hyperparameters."""

        pol = cvx.SinglePeriodOptimization(GammaRisk() * cvx.FullCovariance())

        res = pol.collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 1)

        pol = cvx.SinglePeriodOptimization(
            - GammaRisk() * cvx.FullCovariance()
            - GammaTrade() * cvx.TransactionCost())

        res = pol.collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 2)

        pol = cvx.SinglePeriodOptimization(
            -(GammaRisk() + .5 * GammaRisk()) * cvx.FullCovariance()
            - GammaTrade() * cvx.TransactionCost())

        res = pol.collect_hyperparameters()
        print(res)
        self.assertTrue(len(res) == 3)


if __name__ == '__main__':
    unittest.main()
