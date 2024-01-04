Constraints
===========
	
.. py:module:: cvxportfolio
    :noindex:
    
.. automodule:: cvxportfolio.constraints

.. autoclass:: DollarNeutral

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

.. autoclass:: MaxBenchmarkDeviation

.. autoclass:: MinBenchmarkDeviation

.. autoclass:: ParticipationRateLimit

.. autoclass:: TurnoverLimit



Soft constraints
----------------

(Almost) all the constraints described above can also be be made "soft".
The concept is developed at pages 37-38 of 
`the book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_,
and implemented by the objective term :class:`cvxportfolio.SoftConstraint`,
documented in :ref:`the objective terms page <objective-terms-page>`.

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

In the book we describe having different penalizers for
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


.. code-block:: python

    import cvxportfolio as cvx

    objective = cvx.ReturnsForecast() - gamma_risk * cvx.FullCovariance()

    # for large enough value of the penalizer the constraint becomes "hard"
    objective -= gamma_longonly * cvx.SoftConstraint(cvx.LongOnly())

    cvx.MarketSimulator(universe).backtest(
        cvx.SinglePeriodOptimization(objective).plot()

Some constraint objects, which are in reality compositions of constraints, 
can not be used as soft constraints. See their documentation for more details.

Cost inequality as constraint
-----------------------------

Since version ``0.4.6`` you can use any term described in 
:ref:`the objective terms page <objective-terms-page>` as part 
of an inequality constraint. 
In fact, you can use any linear combination of objective terms.
For a minimal example see the following risk-constrained policy.

.. code-block:: python

    import cvxportfolio as cvx

    # limit the covariance 
    risk_limit = cvx.FullCovariance() <= target_volatility**2

    cvx.MarketSimulator(universe).backtest(
        cvx.SinglePeriodOptimization(cvx.ReturnsForecast(), [risk_limit])).plot()

Keep in mind that the resulting inequality constraint must be convex.
You can't, for example, require that a risk term is larger or equal than
some value.

Base classes (for extending Cvxportfolio)
-----------------------------------------

.. autoclass:: Constraint

.. autoclass:: EqualityConstraint

.. autoclass:: InequalityConstraint





