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

Case-Shiller multi-period
=========================

.. automodule:: examples.case_shiller

.. literalinclude:: ../../examples/case_shiller.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

This is the output printed to screen when executing this script. You can see
many statistics of the back-tests.

.. literalinclude:: ../_static/case_shiller_output.txt
   :language: text

And these are the figure that are plotted. 
The result of the :class:`cvxportfolio.Uniform` policy, which allocates equal
weight to all non-cash assets:

.. figure:: ../_static/case_shiller_uniform.png
   :scale: 100 %
   :alt: case_shiller.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.

And result of the :class:`cvxportfolio.MultiPeriodOptimization` policy, selected
among the efficient frontier below as the one with highest back-tested profit:

.. figure:: ../_static/case_shiller_highest_profit.png
   :scale: 100 %
   :alt: case_shiller.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.

And finally, the efficient frontier, which shows that the 
:class:`cvxportfolio.MultiPeriodOptimization` policies out-perform the
:class:`cvxportfolio.Uniform` allocation in both risk and reward, including
the transaction costs.

  .. figure:: ../_static/case_shiller_frontier.png
     :scale: 100 %
     :alt: hello_world.py result figure

     Efficient frontier of back-test results, which include transaction costs.
