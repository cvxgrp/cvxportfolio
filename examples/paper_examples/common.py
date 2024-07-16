# Copyright (C) 2023-2024 Enzo Busseti
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
"""This module defines some common operations used by the examples."""
from pathlib import Path

import pandas as pd

import cvxportfolio as cvx


def paper_risk_model():
    """Here we build the low-rank plus diagonal risk model used in the paper."""

def paper_returns_forecast():
    """Obtain the returns forecast used by the paper's examples.

    :returns: Returns forecast (synthetic) developed for the paper.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        Path(__file__).parent / 'return_estimate.csv.gz', index_col=0,
        parse_dates=[0])

def paper_simulated_tcost_model():
    """Build the transaction cost model used in simulation.

    This was done in the original example separately in each notebook,
    we factor it here for simplicity.

    We use here some keyword arguments introduced in Cvxportfolio ``1.2.0``,
    which more closely match the original implementation.

    :returns: Transaction cost model to use in simulation.
    :rtype: cvx.costs.SimulatorCost
    """
    sigmas = pd.read_csv(
        Path(__file__).parent / 'sigmas.csv.gz', index_col=0,
        parse_dates=[0])

    return cvx.TcostModel(a=0.0005/2., b=1, sigma=sigmas, exponent=1.5)


def paper_optimization_tcost_model():
    """Build the transaction cost model used in optimization.

    This was done in the original example separately in each notebook,
    we factor it here for simplicity.

    We use here some keyword arguments introduced in Cvxportfolio ``1.2.0``,
    which more closely match the original implementation.

    :returns: Transaction cost model to use in optimization.
    :rtype: cvx.costs.SimulatorCost
    """
    sigma_estimate = pd.read_csv(
        Path(__file__).parent / 'sigma_estimate.csv.gz', index_col=0,
        parse_dates=[0])

    volume_estimate = pd.read_csv(
        Path(__file__).parent / 'volume_estimate.csv.gz', index_col=0,
        parse_dates=[0])

    return cvx.TcostModel(
        a=0.0005/2., b=1, volume_hat=volume_estimate, sigma=sigma_estimate,
        exponent=1.5)

def paper_hcost_model():
    """Build the holding cost model used in simulation and optimization.

    This was done in the original example separately in each notebook,
    we factor it here for simplicity.

    :returns: Holding cost model to use in simulation and optimization.
    :rtype: cvx.costs.SimulatorCost
    """
    # Short fees, in annualized percent (this is equivalent to 1bp per period,
    # as it was in the notebooks).
    borrow_fee = 2.552
    return cvx.HcostModel(short_fees=borrow_fee)
