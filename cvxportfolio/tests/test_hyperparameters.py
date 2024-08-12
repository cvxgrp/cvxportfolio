# Copyright (C) 2023-2024 Enzo Busseti
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
"""Unit tests for the hyper-parameters interface."""

import unittest

import cvxportfolio as cvx
from cvxportfolio.tests import CvxportfolioTest

GAMMA_RISK_RANGE = (.5, 1., 2., 5., 10.)
GAMMA_COST_RANGE = (0., .1, .2, .5, 1., 2., 5., 10.)

class GammaRisk(cvx.RangeHyperParameter):
    """Multiplier of a risk term."""

    def __init__(self, values_range=GAMMA_RISK_RANGE, current_value=1.):
        super().__init__(values_range, current_value)


class GammaTrade(cvx.RangeHyperParameter):
    """Multiplier of a transaction cost term."""

    def __init__(self, values_range=GAMMA_COST_RANGE, current_value=1.):
        super().__init__(values_range, current_value)


class TestHyperparameters(CvxportfolioTest):
    """Test hyper-parameters interface."""

    def test_copy_keeping_multipliers(self):
        """Test that HPs are not duplicated in MPO."""

        pol = cvx.MultiPeriodOpt(
            cvx.ReturnsForecast()
            - cvx.Gamma() * cvx.FullCovariance()
            - cvx.Gamma() * cvx.TransactionCost(),
            planning_horizon=2)
        hps = pol.collect_hyperparameters()

        # TODO make this work, see comment in Estimator.collect_hyperpar
        # self.assertEqual(len(hps), 2)

        self.assertEqual(len(set(hps)), 2)

    def test_repr(self):
        """Test the repr method."""
        obj = cvx.ReturnsForecast() - cvx.Gamma() * cvx.FullCovariance()\
            - cvx.Gamma() * cvx.StocksTransactionCost()

        # print()
        # print(obj)
        # print()

        ref = ('ReturnsForecast(r_hat=HistoricalMeanReturn(half_life=inf,'
            + ' rolling=inf), decay=1.0)'
            + ' - Gamma(current_value=1.0) * FullCovariance('
            + 'Sigma=HistoricalFactorizedCovariance(half_life=inf,'
            + ' rolling=inf, kelly=True))'
            + ' - Gamma(current_value=1.0) * StocksTransactionCost(a=0.0, '
            + 'b=1.0, market_volumes=VolumeHatOrRealized('
            + 'volume_hat=HistoricalMeanVolume(half_life=inf, '
            + "rolling=Timedelta('365 days 05:45:36'))), "
            + "sigma=HistoricalStandardDeviation(half_life=inf, "
            + "rolling=Timedelta('365 days 05:45:36'), kelly=True), "
            + 'exponent=1.5, pershare_cost=0.005)')

        # print()
        # print(ref)
        # print()

        self.assertTrue(str(obj) == ref)

        str(cvx.Gamma() * cvx.Gamma())
        str(cvx.Gamma() - cvx.Gamma())
        str(cvx.Gamma() - (2 * cvx.Gamma()))
        str(cvx.Gamma() - (-2 * cvx.Gamma()))
        str(cvx.Gamma() - (cvx.Gamma() * -2))
        str(cvx.Gamma() - (cvx.Gamma() * -2))
        str(cvx.Gamma() * -1 * (cvx.Gamma() * -2))
        str(2 * cvx.Gamma() - 3 * cvx.Gamma())
        str(2 * cvx.Gamma() * ( cvx.ReturnsForecast()
            - 3 * cvx.Gamma() * cvx.FullCovariance()))
        str(cvx.Gamma() * 2 * ( cvx.ReturnsForecast()
            - cvx.Gamma() * (-3) *  cvx.FullCovariance()))

    def test_basic_hyper_parameters(self):
        """Test simple syntax and errors."""
        gamma = GammaRisk(current_value=1)

        self.assertTrue((-gamma).current_value == -1)
        self.assertTrue((gamma * .5).current_value == .5)
        self.assertTrue((.5 * gamma).current_value == .5)
        self.assertTrue((gamma).current_value == 1)

        cvx.SinglePeriodOptimization(GammaRisk() * cvx.FullCovariance())

        with self.assertRaises(TypeError):
            cvx.SinglePeriodOptimization(GammaRisk * cvx.FullCovariance())

        cvx.SinglePeriodOptimization(-GammaRisk() * cvx.FullCovariance())

    def test_range_hyper_parameter(self):
        """Test range hyperparameter."""

        # pylint: disable=protected-access

        gamma = GammaRisk(current_value=1)
        self.assertTrue((gamma).current_value == 1)
        gamma._decrement()
        self.assertTrue((gamma).current_value == .5)
        with self.assertRaises(IndexError):
            gamma._decrement()

        gamma = GammaRisk(current_value=10)
        self.assertTrue((gamma).current_value == 10)
        with self.assertRaises(IndexError):
            gamma._increment()

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
        # print(res)
        self.assertTrue(len(res) == 1)

        pol = cvx.SinglePeriodOptimization(
            - GammaRisk() * cvx.FullCovariance()
            - GammaTrade() * cvx.TransactionCost())

        res = pol.collect_hyperparameters()
        # print(res)
        self.assertTrue(len(res) == 2)

        pol = cvx.SinglePeriodOptimization(
            -(GammaRisk() + .5 * GammaRisk()) * cvx.FullCovariance()
            - GammaTrade() * cvx.TransactionCost())

        res = pol.collect_hyperparameters()
        # print(res)
        self.assertTrue(len(res) == 3)


if __name__ == '__main__':

    unittest.main() # pragma: no cover
