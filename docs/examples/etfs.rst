Wide asset classes ETFs
=======================

.. automodule:: examples.etfs

.. literalinclude:: ../../examples/etfs.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

This is the output printed to screen when executing this script. You can see
many statistics of the back-tests.

.. include:: ../_static/etfs_output.txt
   :literal:

And these are the figure that are plotted.
The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample Sharpe ratio:

.. figure:: ../_static/etfs_largest_sharpe_ratio.png
   :scale: 100 %
   :alt: examples/etfs.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`

The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy
that has the largest out-of-sample growth rate:

.. figure:: ../_static/etfs_largest_growth_rate.png
   :scale: 100 %
   :alt: examples/etfs.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`
