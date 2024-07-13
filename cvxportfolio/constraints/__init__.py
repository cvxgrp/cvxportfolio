# Copyright 2017-2024 Enzo Busseti
# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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
