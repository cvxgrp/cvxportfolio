.. cvxportfolio documentation master file, created by
sphinx-quickstart on Mon Apr 24 13:19:35 2017.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to CVXPortfolio
========================================

CVXPortfolio is a package for simulating and optimizing multi-period investment based on the framework outlined in the paper `Multi-Period Trading via Convex Optimization <http://stanford.edu/~boyd/papers/cvx_portfolio.html>`_.

The simulator is able to simulate the evolution of a portfolio, taking into account asset returns, transaction costs, and holding costs. The package includes simple but reasonable models of transaction cost based on asset bid-ask spread, volume, and volatility.

CVXPortfolio provides functionality for implementing trading strategies using the convex optimization package `CVXPY`_.

The package relies on `Pandas`_ for data handling (e.g., prices, returns, volumes).
Our simple :ref:`examples <examples>` show how `Quandl`_ can be used to import open source financial data, but any other source can be used instead.

CVXPortfolio is released under a permissive open source :ref:`license <lic>`. It includes basic functionality for simulation and simple or complex optimization based trading. Users can easily extend the package with additional trading strategies.

CVXPortfolio was designed and implemented by Enzo Busseti and Steven Diamond, with input from Stephen Boyd and the authors of `the paper <http://stanford.edu/~boyd/papers/cvx_portfolio.html>`_.

CVXPortfolio is not quite ready yet, but if you want to jump into the development branch `feel free <https://github.com/cvxgrp/cvxportfolio>`_.

.. toctree::
:hidden:

       install/index
       intro/index
       documentation/index
       examples/index
       citing/index
       license/index


.. _Quandl: https://www.quandl.com
.. _CVXPY: https://www.cvxpy.org/
.. _Pandas: http://pandas.pydata.org
