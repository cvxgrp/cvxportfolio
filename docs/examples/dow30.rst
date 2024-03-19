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

.. include:: ../_static/dow30_output.txt
   :literal:

And these are the figure that are plotted. 
The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample Sharpe ratio:

.. figure:: ../_static/dow30_largest_sharpe_ratio.png
   :scale: 100 %
   :alt: dow30_example.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`
   
The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample growth rate:

.. figure:: ../_static/dow30_largest_growth_rate.png
   :scale: 100 %
   :alt: dow30_example.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`

The result of the :class:`cvxportfolio.Uniform` policy, which allocates equal
weight to all non-cash assets:

.. figure:: ../_static/dow30_uniform.png
   :scale: 100 %
   :alt: dow30_example.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`

Finally, the result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
obtained by automatic hyper-parameter optimization to have largest profit:

.. figure:: ../_static/dow30_hyperparameter_optimized.png
   :scale: 100 %
   :alt: dow30_example.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`
