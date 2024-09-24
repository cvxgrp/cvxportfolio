.. Copyright (C) 2023-2024 Enzo Busseti
.. Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.

.. This file is part of Cvxportfolio.

.. Cvxportfolio is free software: you can redistribute it and/or modify it under
.. the terms of the GNU General Public License as published by the Free Software
.. Foundation, either version 3 of the License, or (at your option) any later
.. version.

.. Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
.. ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
.. FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
.. details.

.. You should have received a copy of the GNU General Public License along with
.. Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.

Constraints
===========
	
.. py:module:: cvxportfolio
    :no-index:
    
.. automodule:: cvxportfolio.constraints

.. autoclass:: DollarNeutral

.. autoclass:: MarketNeutral

.. autoclass:: FactorMaxLimit

.. autoclass:: FactorMinLimit

.. autoclass:: FactorGrossLimit

.. autoclass:: FactorNeutral

.. autoclass:: FixedFactorLoading

.. autoclass:: LeverageLimit

.. autoclass:: LongCash

.. autoclass:: LongOnly

.. autoclass:: MaxWeights

.. autoclass:: MinWeights

.. autoclass:: MaxHoldings

.. autoclass:: MinHoldings

.. autoclass:: MaxTradeWeights

.. autoclass:: MinTradeWeights

.. autoclass:: MaxTrades

.. autoclass:: MinTrades

.. autoclass:: MaxBenchmarkDeviation

.. autoclass:: MinBenchmarkDeviation

.. autoclass:: ParticipationRateLimit

.. autoclass:: TurnoverLimit

.. autoclass:: FixedImbalance

.. autoclass:: NoCash

.. _soft-constraints:

Soft constraints
----------------

(Almost) all the constraints described above can also be be made "soft".
The concept is developed in
:paper:`section 4.6 of the paper <section.4.6>`,
and implemented by the objective term :class:`cvxportfolio.SoftConstraint`.

In the case of a linear equality constraint, 
which can be expressed as :math:`h(x) = 0`, 
the corresponding soft constraint is a cost 
of the form :math:`\gamma \|h(x)\|_1`, where
:math:`\gamma` is the *priority* penalizer.

For a linear inequality constraint
:math:`h(x) \leq 0`, instead, 
the corresponding soft constraint is the cost 
:math:`\gamma \|{(h(x))}_+\|_1`, where 
:math:`{(\cdot )}_+` denotes the positive part 
of each element of :math:`h(x)`.

In the paper we describe having different penalizers for
different elements of each constraint vector (so that
the penalizer :math:`\gamma` is a vector).
In our implementation this is achieved by constructing
multiple soft constraints, each with a scalar :math:`\gamma`
penalizer.

The syntax of our implementation is very simple. We pass a constraint instance to
:class:`cvxportfolio.SoftConstraint`, multiply the term by the penalizer, and
subtract it from the objective function.
For a high value of the penalizer the constraint will be
enforced almost exactly. For a small value, it will be almost ignored.

For example:

.. code-block:: python

    policy = cvx.SinglePeriodOptimization(
        objective =
            cvx.ReturnsForecast()
            - 0.5 * cvx.FullCovariance()
            - 10 * cvx.SoftConstraint(cvx.LeverageLimit(3)))

is a policy that almost enforces a leverage limit of 3, allowing for some
violation. This can be controlled by tuning the multiplier in front
of :class:`cvxportfolio.SoftConstraint`.

Some constraint objects, which are in reality compositions of constraints, 
can not be used as soft constraints. See their documentation for more details.

Cost inequality as constraint
-----------------------------

Since version ``0.4.6`` you can use any objective function term,
such as :doc:`returns`, :doc:`risks`, and :doc:`costs`, as part
of an inequality constraint. 
In fact, you can use any linear combination of objective terms.
For a minimal example see the following risk-constrained policy.

.. code-block:: python

    import cvxportfolio as cvx

    # limit the covariance 
    risk_limit = cvx.FullCovariance() <= target_volatility**2

    # or, since Cvxportfolio 1.4.0
    risk_limit_annualized = cvx.FullCovariance() <= cvx.AnnualizedVolatility(
        0.05) # means 5% annualized

    cvx.MarketSimulator(universe).backtest(
        cvx.SinglePeriodOptimization(
            cvx.ReturnsForecast(), [risk_limit])).plot()

Keep in mind that the resulting inequality constraint must be convex.
You can't, for example, require that a risk term is larger or equal than
some value.

Base classes (for extending Cvxportfolio)
-----------------------------------------

.. autoclass:: Constraint

.. autoclass:: EqualityConstraint

.. autoclass:: InequalityConstraint





