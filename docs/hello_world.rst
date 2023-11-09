Hello World Example
===================

This is a simple example that showcases the main usage of the library.
We define a market simulator with some stocks, two trading policies
(one simple, and one optimization-based), backtest them in parallel,
and show the results. This example script is 
`available in the repository <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/hello_world.py>`_.

.. literalinclude:: ../examples/hello_world.py
   :language: python

This is the output printed to screen when executing this script. You can see
many statistics of the back-tests. The timestamps of the back-test are the
open times of the New York stock market (9.30am New York time) expressed in UTC.

.. include:: _static/hello_world_output.txt
   :literal:

And these are the figure that are plotted.
The result of the :class:`cvxportfolio.MultiPeriodOptimization` policy:
   
.. figure:: _static/hello_world.png
   :scale: 100 %
   :alt: hello_world.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`

And result of the :class:`cvxportfolio.Uniform` policy, which allocates equal
weight to all non-cash assets:

.. figure:: _static/hello_world_uniform.png
   :scale: 100 %
   :alt: hello_world.py result figure

   This figure is made by the :meth:`plot()` method of :class:`cvxportfolio.BacktestResult`