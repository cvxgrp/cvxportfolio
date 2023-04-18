% cvxportfolio documentation master file, created by
% sphinx-quickstart on Mon Apr 24 13:19:35 2017.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.


# Cvxportfolio Documentation

**Cvxportfolio is currently under development. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3.**

Cvxportfolio is a package for simulating and optimizing multi-period investment based on the framework outlined in the paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).

The simulator is able to simulate the evolution of a portfolio, taking into account asset returns, transaction costs, and holding costs. The package includes simple but reasonable models of transaction cost based on asset bid-ask spread, volume, and volatility.

CVXPortfolio provides functionality for implementing trading strategies using the convex optimization package [CVXPY].

The package relies on [Pandas] for data handling (e.g., prices, returns, volumes).

CVXPortfolio is released under a permissive open source {ref}`license <lic>`. It includes basic functionality for simulation and simple or complex optimization based trading. Users can easily extend the package with additional trading strategies.

CVXPortfolio was designed and implemented by Enzo Busseti and Steven Diamond, with input from Stephen Boyd and the authors of [the paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).

CVXPortfolio is not quite ready yet, but if you want to jump into the development branch [feel free](https://github.com/cvxgrp/cvxportfolio).


## Installation

To install the package
```
pip install cvxportfolio
```

### Testing

To test it

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
poetry run pytest --cov
```

## Hello World

We suggest to start with our basic "[HelloWorld]" example. It sets up
a portfolio optimization problem and simulates a backtest with 5 years
of real market data. It then presents the results.

To run it clone the repository, 

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
cd examples
poetry run python hello_world.py
```

Continue by looking at the other examples,
also in jupyter notebook format,
or go straight to the {ref}`documentation <documentation>`.

[helloworld]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/Hello_World.py


## Releases
With comments from the [git tags](https://github.com/cvxgrp/cvxportfolio/tags).

- [0.1.1](https://pypi.org/project/cvxportfolio/0.1.1/) Last version before 2023 internals change. If you have built code in
  2016-2023 that uses cvxportfolio versions 0.0.X and relies on internal methods you probably want this release. Subsequent ones will change the internal interfaces. User interfaces instead will mostly remain the same. In addition, this version adds a new module data.py (that here is not called by the rest of cvxportfolio) that simplifies getting data.
- 0.0.X Early development versions. Not tagged in git but [distributed on PyPI](https://pypi.org/project/cvxportfolio/).

## Examples

We present a few example applications built with CVXPortfolio.
Some of these have been developed for our [our book](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) (number 3 to 8).

> 1. [HelloWorld]: basic usage of the simulation and (single period) optimization objects.
> 1. [MultiPeriodTCostOptimization]: basic usage of the multi period optimization framework.

The following scripts are currently being restored. They were used to generate the plots and results in [the book](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
> 3. [DataEstimatesRiskModel]: download and clean the data used for the examples in our paper. (Its output files are available in the [data](https://github.com/cvxgrp/cvxportfolio/blob/master/data) folder of the repo.)
> 4. [PortfolioSimulation]: simple simulation of a portfolio rebalanced periodically to a target benchmark.
> 5. [SinglePeriodOptimization]: example of the single period optimization framework, with search of optimal hyper-parameters.
> 6. [MultiPeriodOptimization]: same for the multi period optimization framework.
> 7. [SolutionTime]: analysis of execution time of the simulation and optimization code.
> 8. [RealTimeOptimization]: get a vector of trades to execute in real time (exports to Excel format).

[dataestimatesriskmodel]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/DataEstimatesRiskModel.py
[helloworld]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/hello_world.py
[multiperiodoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/MultiPeriodOptimization.py
[MultiPeriodTCostOptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/multi_period_tcost_optimization.py
[portfoliosimulation]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/PortfolioSimulation.py
[realtimeoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/RealTimeOptimization.py
[singleperiodoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/SinglePeriodOptimization.py
[solutiontime]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/SolutionTime.py



## Publications

CVXPortfolio implements the ideas developed in our accompanying [paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).

If you use CVXPortfolio for published work, please cite it as

```latex
@article{BBDKKNS:17
    author       = {S. Boyd and E. Busseti and S. Diamond and R. Kahn and K. Koh and P. Nystrup and J. Speth},
    title        = {Multi-Period Trading via Convex Optimization},
    journal      = {Foundations and Trends in Optimization},
    year         = {2017},
    pages        = {to appear}
    publisher    = {Now Publishers}
    url          = {http://stanford.edu/~boyd/papers/cvx_portfolio.html},
}
```

## License

Cvxportfolio is licensed under the [Apache 2.0](http://www.apache.org/licenses/) permissive
open source license.




```{toctree}
:hidden: true

Home<self>
documentation/index
Repository <https://github.com/cvxgrp/cvxportfolio>
```

[cvxpy]: https://www.cvxpy.org/
[pandas]: http://pandas.pydata.org

