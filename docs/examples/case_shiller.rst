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

.. include:: ../_static/case_shiller_output.txt
   :literal:

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
