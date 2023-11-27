# [Cvxportfolio](https://www.cvxportfolio.com)

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)



Cvxportfolio is an object-oriented library for portfolio optimization and back-testing. It implements models described in the
[accompanying paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).

The documentation of the library is at [www.cvxportfolio.com](https://www.cvxportfolio.com).

Installation
------------
You can install our latest release with

```
pip install -U cvxportfolio
```
You can see how this works on our [Installation and Hello World](https://youtu.be/1ThOKEu371M) youtube video.

Testing locally
---------------
After installing you can run our unit test suite in you local environment by

```
python -m cvxportfolio.tests
```


Simple Example
----------------
In the following example market data is downloaded by a public source
(Yahoo finance) and the forecasts are computed iteratively, 
at each point in the backtest, from past data. 


```python
import cvxportfolio as cvx

gamma = 3       # risk aversion parameter (Chapter 4.2)
kappa = 0.05    # covariance forecast error risk parameter (Chapter 4.3)
objective = cvx.ReturnsForecast() - gamma * (
	cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost()
constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.StockMarketSimulator(['AAPL', 'AMZN', 'TSLA', 'GM', 'CVX', 'NKE'])

result = simulator.backtest(policy, start_time='2020-01-01')

# print back-test result statistics
print(result)

# plot back-test results
result.plot()
```

At each point in the back-test,
the policy object only operates on **past data**, and thus the result you 
get is a realistic simulation of what the strategy would have performed in 
the market.
Returns are forecasted as the historical mean returns 
and covariances as historical covariances (both ignoring `np.nan`'s).
The simulator by default includes holding and transaction costs, using the
models described in the paper, and default parameters that are typical for the
US stock market.



Some Other Examples
-------------------
We show in the example on [user-provided forecasters](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/user_provided_forecasters.py) 
how the user can define custom classes to forecast
the expected returns and covariances. These provide callbacks that are
executed at each point in time during the back-test. The system enforces 
causality and safety against numerical errors. 
We recommend to always include 
the default forecasters that we provide in any analysis you may do, 
since they are very robust and well-tested. 

We show in the examples on [DOW30 components](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/dow30_example.py) 
and [wide assets-classes ETFs](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/etfs_example.py) how a
simple sweep over hyper-parameters, taking advantage of our sophisticated parallel backtest machinery, 
quickly provides results on the best strategy to apply to any given selection of assets.


Development
-----------
To set up a development environment locally you should clone
the repository (or,
[fork on Github](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
and then clone your fork)

```bash
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
```

Then, you should have a look at our
[Makefile](https://www.gnu.org/software/make/manual/make.html#Introduction)
and possibly change the `PYTHON` variable to match your system's python
interpreter. Once you have done that,

```bash
make env
make test
```

This will replicate our [development environment](https://docs.python.org/3/library/venv.html)
and run our test suite.

You activate the shell environment with one of scripts in `env/bin`
(or `env\Scripts` on Windows), for example if you use bash on POSIX

```bash
source env/bin/activate
```
and from the environment you can run any of the scripts in the examples
(the cvxportfolio package is installed in
[editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)).
Or, if you don't want to activate the environment, you can just run scripts
directly using `env/bin/python` (or `env\Scripts\python` on Windows)
like we do in the Makefile.

Additionally, to match our CI/CD pipeline, you may set the following
[git hooks](https://git-scm.com/docs/githooks)

```bash
echo "make lint" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "make test" > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Examples from the paper
-----------------------
In branch [0.0.X](https://github.com/cvxgrp/cvxportfolio/tree/0.0.X) you can find the original material used to generate plots
and results in the paper. As you may see from those
ipython notebooks a lot of the logic that was implemented there, outside of Cvxportfolio proper, is being included and made automatic
in newer versions of Cvxportfolio. 


Citing
------------

If you use Cvxportfolio in work that leads to publication, you can cite the following:

```
@misc{busseti2017cvx,
    author    = "Busseti, Enzo and Diamond, Steven and Boyd, Stephen",
    title     = "Cvxportfolio",
    month    = "January",
    year     = "2017",
    note     = "Portfolio Optimization and Back--{T}esting",
    howpublished = {\url{https://github.com/cvxgrp/cvxportfolio}},
}

@article{boyd2017multi,
  author  = "Boyd, Stephen and Busseti, Enzo and Diamond, Steven and Kahn, Ron and Nystrup, Peter and Speth, Jan",
  journal = "Foundations and Trends in Optimization",
  title   = "Multi--{P}eriod Trading via Convex Optimization",
  month   = "August",
  year    = "2017",
  number  = "1",
  pages   = "1--76",
  volume  = "3",
  url     = {\url{https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf}},
}
```

The latter is also the first chapter of this thesis:

```
@phdthesis{busseti2018portfolio,
    author    = "Busseti, Enzo",
    title     = "Portfolio Management and Optimal Execution via Convex Optimization",
    school    = "Stanford University",
    address   = "Stanford, California, USA",
    month    = "May",
    year     = "2018",
    url     = {\url{https://stacks.stanford.edu/file/druid:wm743bj5020/thesis-augmented.pdf}},
}
```

License
------------

Cvxportfolio is licensed under the [Apache 2.0](http://www.apache.org/licenses/) permissive
open source license.


