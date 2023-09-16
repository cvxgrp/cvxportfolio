Hello World Example
===================

This is a simple example that showcases the main usage of the library.
We define a market simulator with some stocks, two trading policies
(one simple, and one optimization-based), backtest them in parallel,
and show the results. This example script is 
`available in the repository <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/hello_world.py>`_.

.. literalinclude:: ../examples/hello_world.py
   :language: python

This is the output printed to screen when executing this script.

.. include:: _static/hello_world_output.txt
   :literal:

And this is the figure that is the plotted.
   
.. figure:: _static/hello_world.png
   :scale: 100 %
   :alt: hello_world.py result figure

   This is the figure that is made by the ``plot()`` method of :class:`BacktestResult`