% cvxportfolio documentation master file, created by
% sphinx-quickstart on Mon Apr 24 13:19:35 2017.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# Welcome to CVXPortfolio

**Cvxportfolio is currently under redevelopment. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3.**

CVXPortfolio is a package for simulating and optimizing multi-period investment based on the framework outlined in the paper [Multi-Period Trading via Convex Optimization](http://stanford.edu/~boyd/papers/cvx_portfolio.html).

The simulator is able to simulate the evolution of a portfolio, taking into account asset returns, transaction costs, and holding costs. The package includes simple but reasonable models of transaction cost based on asset bid-ask spread, volume, and volatility.

CVXPortfolio provides functionality for implementing trading strategies using the convex optimization package [CVXPY].

The package relies on [Pandas] for data handling (e.g., prices, returns, volumes).
Our simple {ref}`examples <examples>` show how [Quandl] can be used to import open source financial data, but any other source can be used instead.

CVXPortfolio is released under a permissive open source {ref}`license <lic>`. It includes basic functionality for simulation and simple or complex optimization based trading. Users can easily extend the package with additional trading strategies.

CVXPortfolio was designed and implemented by Enzo Busseti and Steven Diamond, with input from Stephen Boyd and the authors of [the paper](http://stanford.edu/~boyd/papers/cvx_portfolio.html).

CVXPortfolio is not quite ready yet, but if you want to jump into the development branch [feel free](https://github.com/cvxgrp/cvxportfolio).

```{toctree}
:hidden: true

install/index
intro/index
API/index
examples/index
citing/index
license/index
```

[cvxpy]: https://www.cvxpy.org/
[pandas]: http://pandas.pydata.org
[quandl]: https://www.quandl.com
