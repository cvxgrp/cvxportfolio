# Copyright (C) 2017-2024 Enzo Busseti
# Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
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
from .test_result import TestResult
from .test_returns import TestReturns
from .test_risks import TestRisks
from .test_simulator import TestSimulator
from .test_utils import TestUtils

if __name__ == '__main__':

    unittest.main(warnings='error')
