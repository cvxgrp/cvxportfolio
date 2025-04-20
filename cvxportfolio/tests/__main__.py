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
#
## Earlier versions of this module had the following copyright and licensing
## notice, which is subsumed by the above.
##
### Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###    http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
"""Run all tests with ``python -m cvxportfolio.tests``.

We add custom logic to Python's default test runner in order to ignore
``cvx.errors.DownloadError`` with a command line switch.
"""

# pylint: disable=unused-import, too-few-public-methods

import sys
import unittest

from ..errors import DownloadError
from .test_constraints import TestConstraints
from .test_costs import TestCosts
from .test_data import TestData, TestMarketData
from .test_estimator import TestEstimator
from .test_forecast import TestForecast
from .test_hyperparameters import TestHyperparameters
from .test_integration import TestIntegration
from .test_policies import TestPolicies
from .test_result import TestResult
from .test_returns import TestReturns
from .test_risks import TestRisks
from .test_simulator import TestSimulator
from .test_utils import TestUtils


class TextTestResultAllowDownloadError(unittest.runner.TextTestResult):
    """Test result customized to ignore cvx.errors.DownloadError."""

    def addError( # pylint: disable=missing-param-doc,missing-type-doc
            self, test, err):
        """Called when an error has occurred, ignore DownloadError."""
        if err[0] is not DownloadError:
            super().addError(test, err)
        else:
            print('\n' + '#'*79)
            print(f'TEST {test} threw cvx.errors.DownloadError, ignoring!')
            print(f'Exception details: {err}')
            print('#'*79)

class TextTestRunnerAllowDownloadError(unittest.runner.TextTestRunner):
    """Test runner customized to ignore cvx.errors.DownloadError."""
    resultclass = TextTestResultAllowDownloadError

class mainOptionallyAllowDownloadError(  # pylint: disable=invalid-name
        unittest.main):
    """Add switch to main's argparse to allow cvx.errors.DownloadError."""

    def _getParentArgParser(self):
        """Add switch to parser."""
        parser = super()._getParentArgParser()
        parser.add_argument(
            '--ignore-download-errors', action='store_true',
            help='Ignore cvx.errors.DownloadError thrown by tests.')
        return parser

    def parseArgs( # pylint: disable=missing-param-doc,missing-type-doc
            self, argv):
        """Parse command line arguments including our custom switch."""
        super().parseArgs(argv)
        if self._main_parser.parse_args().ignore_download_errors:
            self.testRunner = TextTestRunnerAllowDownloadError

if __name__ == '__main__': # pragma: no cover
    if sys.version_info.minor > 9:
        # DeprecationWarning's may be thrown
        # when running with old versions
        mainOptionallyAllowDownloadError(warnings='error')
    else:
        mainOptionallyAllowDownloadError()
