# Copyright 2023 Enzo Busseti
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
