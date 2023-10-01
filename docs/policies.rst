.. _policies-page:

Trading policies
================

.. automodule:: cvxportfolio.policies

.. py:module:: cvxportfolio
    :noindex:

.. _optimization-policies-page:

Optimization-based policies
---------------------------

.. autoclass:: SinglePeriodOptimization

    .. automethod:: execute

.. autoclass:: MultiPeriodOptimization

    .. automethod:: execute


Simple policies
---------------

.. autoclass:: AdaptiveRebalance

    .. automethod:: execute

.. autoclass:: Hold

    .. automethod:: execute
    
.. autoclass:: AllCash

    .. automethod:: execute

.. autoclass:: FixedTrades

    .. automethod:: execute

.. autoclass:: FixedWeights

    .. automethod:: execute

.. autoclass:: PeriodicRebalance

    .. automethod:: execute

.. autoclass:: ProportionalRebalance

    .. automethod:: execute

.. autoclass:: ProportionalTradeToTargets

    .. automethod:: execute

.. autoclass:: RankAndLongShort

    .. automethod:: execute

.. autoclass:: SellAll

    .. automethod:: execute
    
.. autoclass:: MarketBenchmark

    .. automethod:: execute

Base policy class (for defining your own policy)
------------------------------------------------

.. autoclass:: cvxportfolio.policies.Policy

