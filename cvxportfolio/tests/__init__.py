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
"""We make the tests a sub-package so we can ship them."""

import logging
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cvxpy as cp
import numpy as np
import pandas as pd

import cvxportfolio as cvx

logger = logging.getLogger()

class CvxportfolioTest(unittest.TestCase):
    """Base class for Cvxportfolio unit tests."""

    @classmethod
    def setUpClass(cls):
        """Initialize test class."""
        # gets deleted automatically when de-referenced
        # pylint: disable=consider-using-with
        cls._tempdir = TemporaryDirectory()
        cls.datadir = Path(cls._tempdir.name)
        logger.info('using TemporaryDirectory %s', cls.datadir)

        cls.sigma = pd.read_csv(
            Path(__file__).parent / "sigmas.csv",
            index_col=0, parse_dates=[0])
        cls.returns = pd.read_csv(
            Path(__file__).parent /
            "returns.csv", index_col=0, parse_dates=[0])
        cls.volumes = pd.read_csv(
            Path(__file__).parent /
            "volumes.csv", index_col=0, parse_dates=[0])
        cls.prices = pd.DataFrame(
            np.random.uniform(10, 200, size=cls.volumes.shape),
            index=cls.volumes.index, columns=cls.volumes.columns)
        cls.market_data = cvx.UserProvidedMarketData(
            returns=cls.returns, volumes=cls.volumes, prices=cls.prices,
            cash_key='cash',
            min_history=pd.Timedelta('0d'))
        cls.universe = cls.returns.columns
        cls.w_plus = cp.Variable(cls.returns.shape[1])
        cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
        cls.z = cp.Variable(cls.returns.shape[1])
        cls.N = cls.returns.shape[1]

        # with this we make sure we don't keep asking YahooFinance and Fred
        # for new data during the test execution
        cls.data_grace_period = pd.Timedelta('10d')

        # working around CVXPY-ECOS deprecation issues
        try:
            import ecos as _  # pylint: disable=import-outside-toplevel
            cls.default_socp_solver = 'ECOS'
        except ImportError: # pragma: no cover
            cls.default_socp_solver = None # lets CVXPY choose

        # not necessary for now, but good to control
        cls.default_qp_solver = 'OSQP'
        cls.timers = {}

    @classmethod
    def tearDownClass(cls):
        """Finalize test class."""
        print('Timing report:')
        print(pd.Series(cls.timers))

    @staticmethod
    def strip_tz_and_hour(market_data):
        """Transform DFs indexes from datetime with tz to date without tz.

        :param market_data: Market data object whose dataframes get modified.
        :type market_data: :class:`cvxportfolio.data.MarketData`
        """
        market_data.returns.index = \
            market_data.returns.index.tz_localize(None).floor("D")
        market_data.volumes.index = \
            market_data.volumes.index.tz_localize(None).floor("D")
        market_data.prices.index = \
            market_data.prices.index.tz_localize(None).floor("D")

    def boilerplate(self, model):
        """Initialize objects, compile cvxpy expression.

        :param model: Model to compile (constraint, objective term, ...).
        :type model: :class:`cvxportfolio.CvxpyEstimator`

        :returns: Compiled Cvxpy object.
        :rtype: Expression, constraint, or list of constraints.
        """
        model.initialize_estimator_recursive(
            universe=self.returns.columns,
            trading_calendar=self.returns.index)
        return model.compile_to_cvxpy(
            w_plus=self.w_plus, z=self.z,
            w_plus_minus_w_bm=self.w_plus_minus_w_bm)

    def setUp(self):
        """Timer for each test."""
        self.start_time = time.time()

    def tearDown(self):
        """Save timer for each test."""
        t = time.time() - self.start_time
        self.timers[str(self.id())] = t

    def _difficult_market_data(self):
        """Market data with difficult universe changes.

        Used for BacktestResult correct handling of IPOs/delistings.
        """
        rets = pd.DataFrame(self.returns.iloc[:, -10:], copy=True)
        volumes = pd.DataFrame(self.volumes.iloc[:, -9:], copy=True)
        prices = pd.DataFrame(self.prices.iloc[:, -9:], copy=True)
        rets.iloc[15:25, 1:3] = np.nan
        rets.iloc[9:17, 3:5] = np.nan
        rets.iloc[8:15, 5:7] = np.nan
        rets.iloc[17:29, 7:8] = np.nan
        # print(rets.iloc[10:20])
        with self.assertLogs(level='WARNING'):
            modified_market_data = cvx.UserProvidedMarketData(
                returns=rets, volumes=volumes, prices=prices,
                cash_key='cash',
                min_history=pd.Timedelta('0d'))
        t_start = rets.index[10]
        t_end = rets.index[20]
        return modified_market_data, t_start, t_end

    def _difficult_simulator_and_policies(self):
        """Get difficult simulator and policies.

        Used for object re-use tests.
        """
        md, t_s, t_e = self._difficult_market_data()
        simulator = cvx.StockMarketSimulator(
            market_data=md, base_location=self.datadir)
        policy1 = cvx.Uniform()
        policy2 = cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast() - .5 * cvx.FullCovariance()
                - cvx.StocksTransactionCost(),
            [cvx.LongOnly(applies_to_cash=True)], planning_horizon=2)
        policy3 = cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast() - .5 * cvx.FullCovariance(),
            [cvx.LongOnly(applies_to_cash=True)],
            planning_horizon=2, solver='OSQP')
            # because OSQP allocates a C struct that can't be pickled
        return simulator, t_s, t_e, (policy1, policy2, policy3)
