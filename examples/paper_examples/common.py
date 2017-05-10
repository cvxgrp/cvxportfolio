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
"""This module defines some common operations used by the examples."""
from pathlib import Path

import pandas as pd

import cvxportfolio as cvx


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