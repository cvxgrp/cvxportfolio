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

DOW30 monthly
=============

.. automodule:: examples.dow30

.. literalinclude:: ../../examples/dow30.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

This is the output printed to screen when executing this script. You can see
many statistics of the back-tests.

.. literalinclude:: ../_static/dow30_output.txt
   :language: text

And these are the figure that are plotted.
The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample Sharpe ratio:

.. figure:: ../_static/dow30_largest_sharpe_ratio.png
   :scale: 100 %
   :alt: examples/dow30.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.

The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample growth rate:

.. figure:: ../_static/dow30_largest_growth_rate.png
   :scale: 100 %
   :alt: examples/dow30.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.

The result of the :class:`cvxportfolio.Uniform` policy, which allocates equal
weight to all non-cash assets:

.. figure:: ../_static/dow30_uniform.png
   :scale: 100 %
   :alt: examples/dow30.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.

Finally, the result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
obtained by automatic hyper-parameter optimization to have largest profit:

.. figure:: ../_static/dow30_hyperparameter_optimized.png
   :scale: 100 %
   :alt: examples/dow30.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.
