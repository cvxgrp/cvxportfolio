# Copyright (C) 2024 Enzo Busseti
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
"""Assorted integration tests.

TODO: Many tests that are in ``test_simulator.py`` could be moved here.
"""

import unittest

import numpy as np

import cvxportfolio as cvx
from cvxportfolio.tests import CvxportfolioTest


class TestIntegration(CvxportfolioTest):
    """Assorted integration tests."""

    def test_holdings_limit(self):
        """Test Max/MinHoldings."""

        md, start, end = self._difficult_market_data()
        sim = cvx.MarketSimulator(market_data=md, base_location=self.datadir)
        pol = cvx.SinglePeriodOpt(
            cvx.ReturnsForecast() - 5 * cvx.FullCovariance(),
            [cvx.MinHoldings(-1000), cvx.MaxHoldings(1000)]
        )
        result = sim.backtest(pol, start_time=start, end_time=end)
        self.assertTrue(
            np.all(result.h_plus.iloc[:, :-1].fillna(0.) - 1 < 1000))
        self.assertTrue(
            np.all(result.h_plus.iloc[:, :-1].fillna(0.) + 1 > -1000))
        # print(result.h_plus)

    def test_trades_limit(self):
        """Test Max/MinTrades."""

        md, start, end = self._difficult_market_data()
        sim = cvx.MarketSimulator(market_data=md, base_location=self.datadir)
        pol = cvx.SinglePeriodOpt(
            cvx.ReturnsForecast() - 5 * cvx.FullCovariance(),
            [cvx.MinTrades(-1234), cvx.MaxTrades(9876)]
        )
        result = sim.backtest(pol, start_time=start, end_time=end)
        self.assertTrue(
            np.all(result.u.iloc[:, :-1].fillna(0.) - 1 < 9876))
        self.assertTrue(
            np.all(result.u.iloc[:, :-1].fillna(0.) + 1 > -1234))
        # print(result.u)

if __name__ == '__main__': # pragma: no cover
    unittest.main(warnings='error')
