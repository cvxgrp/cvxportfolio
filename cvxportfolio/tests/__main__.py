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
"""Run all tests with ``python -m cvxportfolio.tests``."""

import unittest

# pylint: disable=[unused-import]
from .test_constraints import TestConstraints
from .test_costs import TestCosts
from .test_data import TestData, TestMarketData
from .test_estimator import TestEstimator
from .test_forecast import TestForecast
from .test_hyperparameters import TestHyperparameters
from .test_policies import TestPolicies
from .test_returns import TestReturns
from .test_risks import TestRisks
from .test_simulator import TestSimulator
from .test_utils import TestUtils

if __name__ == '__main__':

    unittest.main()#warnings='error')
