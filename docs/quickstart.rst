Quickstart
==========

Cvxportfolio is designed for ease of use and extension. With it, you can
quickly experiment with financial trading strategies and develop new ones.

For example, a classic Markowitz optimization strategy 

.. math::

    \begin{array}{ll}
         \text{maximize} & w^T \mathbf{E}[r] - \frac{1}{2} w^T \mathbf{Var}[r] w \\
         \text{subject to} & w \geq 0, w^T \mathbf{1} <= 1
    \end{array}

is specified as follows

.. code-block:: python
    
    import cvxportfolio as cvx
    
    objective = cvx.ReturnsForecast() - 0.5 * cvx.FullCovariance()
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
    
    strategy = cvx.SinglePeriodOptimization(objective, constraints)
    
Here we instantiated an :ref:`optimization-based trading policy <optimization-policies-page>`.
Cvxportfolio defines two (the other is a simple extension of this). 
Optimization-based policies are defined by an objective function, which is 
maximized, and a list of constraints, that are imposed on the solution.
The objective function is specified as a linear combination of simple
terms, :doc:`we provide many <objective_terms>`, and it's easy to define new ones.
We provide as well :doc:`many constraints <constraints>` to choose from, and
it's even easier to define new ones.

Where are the assets' names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cvxportfolio policies are symbolic, they only define the logic of a trading 
strategy. They work with any selection of assets. 
The two objective terms defined above, however, need to know the values of
the expected returns and the covariance matrix. 
You can :ref:`pass Pandas dataframes <passing-data>` for those, and in that
case your dataframes should contain the assets' names you want to trade. 
Or, if you want, or you can rely on :doc:`forecasters <forecasts>` to compute those
(iteratively, in back-testing) using past data. That is what happens in the 
code shown above, the default parameters of :class:`cvxportfolio.ReturnsForecast`
and :class:`cvxportfolio.FullCovariance` are forecasters that compute historical
means and historical covariances respectively, at each point in time if running in a back-test
or once, if running live, by using the policy's :meth:`execute` method.

*To be continued.*
