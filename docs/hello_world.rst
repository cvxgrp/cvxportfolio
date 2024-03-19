Hello World Example
===================

.. automodule:: examples.hello_world

.. literalinclude:: ../examples/hello_world.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

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