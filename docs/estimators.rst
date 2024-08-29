.. Copyright (C) 2023-2024 Enzo Busseti

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

Estimators
===========

.. automodule:: cvxportfolio.estimator

.. py:module:: cvxportfolio.estimator
     :no-index:

.. autoclass:: Estimator

     .. automethod:: initialize_estimator

     .. automethod:: initialize_estimator_recursive

     .. automethod:: values_in_time

     .. automethod:: values_in_time_recursive

     .. automethod:: finalize_estimator

     .. automethod:: finalize_estimator_recursive

.. autoclass:: SimulatorEstimator

     .. automethod:: simulate

     .. automethod:: simulate_recursive

.. autoclass:: CvxpyExpressionEstimator

     .. automethod:: compile_to_cvxpy

.. autoclass:: DataEstimator

