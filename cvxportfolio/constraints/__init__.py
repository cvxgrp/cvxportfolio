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
"""Here we define many realistic constraints that apply to :ref:`portfolio
optimization trading policies <optimization-policies-page>`.

Some of them, like :class:`LongOnly`, are
very simple to use. Some others are more advanced,
for example :class:`FactorNeutral`
takes time-varying factor exposures as parameters.

For a minimal example we present the classic Markowitz allocation.

.. code-block:: python

    import cvxportfolio as cvx

    objective = cvx.ReturnsForecast() - gamma_risk * cvx.FullCovariance()

    # the policy takes a list of constraint instances
    constraints = [cvx.LongOnly(applies_to_cash=True)]

    policy = cvx.SinglePeriodOptimization(objective, constraints)
    print(cvx.MarketSimulator(universe).backtest(policy))

With this, we require that the optimal post-trade weights
found by the single-period optimization policy are non-negative.
In our formulation the full portfolio weights vector (which includes
the cash account) sums to one,
see equation 4.9 at :paper:`page 43 of the paper <section.4.8>`.
"""

from .base_constraints import *
from .constraints import *
