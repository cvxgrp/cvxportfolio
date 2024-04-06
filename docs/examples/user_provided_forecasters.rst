User-provided forecasters
=========================

.. automodule:: examples.user_provided_forecasters

.. literalinclude:: ../../examples/user_provided_forecasters.py
   :language: python
   :start-after: if __name__ ==
   :end-before: # we use this to save the plots
   :dedent:

This is the output printed to screen when executing this script. If you compare
to a back-test using the standard Cvxportfolio forecasters you may notice
faster execution time of the policy object; that is because Cvxportfolio
default forecasters are optimized for sequential evaluation, at each point in
time of a back-test they don't necessarily compute the forecasts from scratch,
but update the ones computed at the period before (if possible).

.. include:: ../_static/user_provided_forecasters_output.txt
   :literal:

And this is the figure that is plotted. You can see that, compared to the
standard Cvxportfolio forecasts (which use all available historical data at
each point in time) this back-test has a much less stable allocation, and
changes much more with the market conditions.

.. figure:: ../_static/user_provided_forecasters.png
   :scale: 100 %
   :alt: examples/user_provided_forecasters.py result figure

   This figure is made by the :meth:`cvxportfolio.result.BacktestResult.plot` method.
