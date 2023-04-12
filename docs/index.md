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
poetry install
poetry run pytest --cov
```

## Hello World

We suggest to start with our basic "[HelloWorld]" example. It sets up
a portfolio optimization problem and simulates a backtest with 5 years
of real market data. It then presents the results.

To run it clone the repository, create the environment and the `cvxportfolio` kernel, then start `jupyter`

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
bash create_kernel.sh
cd examples
jupyter notebook
```

Continue by looking at the other examples,
also in jupyter notebook format,
or go straight to the {ref}`documentation <documentation>`.

[helloworld]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/HelloWorld.ipynb
[jupyter]: https://jupyter.org/


## Examples

We present a few example applications built with CVXPortfolio.
Some of these have been developed for our [paper](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) (number 3 to 7).

> 1. [HelloWorld]: basic usage of the simulation and (single period) optimization objects.
> 1. [MultiPeriodTCostOptimization]: basic usage of the multi period optimization framework.

The following notebooks are currently (2023-04-11) broken and are being restored. They were used to generate the plots and results in [the paper](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
> 3. [DataEstimatesRiskModel]: download and clean the data used for the examples in our paper. (Its output files are available in the [data](https://github.com/cvxgrp/cvxportfolio/blob/master/data) folder of the repo.)
> 4. [PortfolioSimulation]: simple simulation of a portfolio rebalanced periodically to a target benchmark.
> 5. [SinglePeriodOptimization]: example of the single period optimization framework, with search of optimal hyper-parameters.
> 6. [MultiPeriodOptimization]: same for the multi period optimization framework.
> 7. [SolutionTime]: analysis of execution time of the simulation and optimization code.
> 8. [RealTimeOptimization]: get a vector of trades to execute in real time (exports to Excel format).

[dataestimatesriskmodel]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/DataEstimatesRiskModel.ipynb
[helloworld]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/HelloWorld.ipynb
[multiperiodoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/MultiPeriodOptimization.ipynb
[MultiPeriodTCostOptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/MultiPeriodTCostOptimization.ipynb
[portfoliosimulation]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/PortfolioSimulation.ipynb
[realtimeoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/RealTimeOptimization.ipynb
[singleperiodoptimization]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/SinglePeriodOptimization.ipynb
[solutiontime]: https://github.com/cvxgrp/cvxportfolio/blob/master/examples/SolutionTime.ipynb



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

