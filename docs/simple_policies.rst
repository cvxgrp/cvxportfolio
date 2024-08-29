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

Simple policies
---------------

.. py:module:: cvxportfolio
    :no-index:

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

.. autoclass:: MarketBenchmark

    .. automethod:: execute

.. autoclass:: PeriodicRebalance

    .. automethod:: execute

.. autoclass:: ProportionalRebalance

    .. automethod:: execute

.. autoclass:: ProportionalTradeToTargets

    .. automethod:: execute

.. autoclass:: RankAndLongShort

    .. automethod:: execute

.. autoclass:: Uniform

    .. automethod:: execute
    
.. autoclass:: SellAll