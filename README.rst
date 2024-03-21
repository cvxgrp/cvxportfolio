`Cvxportfolio <https://www.cvxportfolio.com>`__
===============================================

|CVXportfolio on PyPI| |linting: pylint| |Coverage Status|
|Documentation Status| |Apache 2.0 License| |Anaconda-Server Badge|


Cvxportfolio is an object-oriented library for portfolio optimization
and back-testing. It implements models described in the `accompanying paper
<https://cvxportfolio.readthedocs.org/en/master/_static/cvx_portfolio.pdf>`_.

The documentation of the library is at
`www.cvxportfolio.com <https://www.cvxportfolio.com>`__.

.. Installation

*News:*

   Since end of 2023 we're running daily `example strategies
   <https://github.com/cvxgrp/cvxportfolio/tree/master/examples/strategies>`_
   using the development version (master branch); each day we commit target
   weights and initial holdings to the repository. All the code that runs them,
   including the cron script, is in the repository.

Installation
------------

Cvxportolio is written in Python and can be easily installed in any Python
environment by simple:

.. code:: bash

   pip install -U cvxportfolio

You can see how this works on our `Installation and Hello
World <https://youtu.be/1ThOKEu371M>`__ youtube video. 
Anaconda installs 
`are also supported <https://anaconda.org/conda-forge/cvxportfolio>`_.

Cvxportfolio's main dependencies are `Cvxpy <https://www.cvxpy.org>`_ for
interfacing with numerical solvers and `Pandas <https://pandas.pydata.org/>`_
for interfacing with databases. We don't require any specific version of our
dependencies and test against all recent ones (up to a few years ago).


.. Test

Test
----

After installing you can run our unit test suite in you local environment by

.. code:: bash

   python -m cvxportfolio.tests

We test against recent python versions (3.8, 3.9, 3.10, 3.11, 3.12) and recent versions
of the main dependencies (from pandas 1.4, cvxpy 1.1, ..., up to the current
versions) on all major operating systems. You can see the `automated testing code 
<https://github.com/cvxgrp/cvxportfolio/blob/master/.github/workflows/test.yml>`_.


.. Simple Example

Simple example
--------------

In the following example market data is downloaded by a public source
(Yahoo finance) and the forecasts are computed iteratively, at each
point in the backtest, from past data.

.. code:: python

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

At each point in the back-test, the policy object only operates on
**past data**, and thus the result you get is a realistic simulation of
what the strategy would have performed in the market. Returns are
forecasted as the historical mean returns and covariances as historical
covariances (both ignoring ``np.nan``\ ’s). The simulator by default
includes holding and transaction costs, using the models described in
the paper, and default parameters that are typical for the US stock
market.

Other examples
--------------

`Many examples 
<https://www.cvxportfolio.com/en/stable/examples.html>`_
are shown in the documentation website, along with
their output and comments.

`Even more example scripts
<https://github.com/cvxgrp/cvxportfolio/blob/master/examples>`_ 
are available in the code repository. 

`The original examples from the paper 
<https://github.com/cvxgrp/cvxportfolio/tree/0.0.X/examples>`_ 
are visible in a dedicated branch,
and are being translated to run with the stable versions (``1.0.0`` and above) of the
library. The translations are visible at `this documentation page
<https://www.cvxportfolio.com/en/stable/examples/paper_examples.html>`_.

We show in the example on `user-provided
forecasters <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/user_provided_forecasters.py>`__
how the user can define custom classes to forecast the expected returns
and covariances. These provide callbacks that are executed at each point
in time during the back-test. The system enforces causality and safety
against numerical errors. We recommend to always include the default
forecasters that we provide in any analysis you may do, since they are
very robust and well-tested.

We show in the examples on `DOW30
components <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/dow30_example.py>`__
and `wide assets-classes
ETFs <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/etfs_example.py>`__
how a simple sweep over hyper-parameters, taking advantage of our
sophisticated parallel backtest machinery, quickly provides results on
the best strategy to apply to any given selection of assets.

Similar projects
----------------

There are many open-source projects for portfolio optimization and back-testing.
Some notable ones in the Python ecosystem are `Zipline <https://github.com/quantopian/zipline>`_,
which implements a call-back model for back-testing very similar to the one
we provide, `Riskfolio-Lib <https://riskfolio-lib.readthedocs.io/en/latest/examples.html>`_
which implements (many!) portfolio optimization models and also follows a modular
approach like ours, `VectorBT <https://vectorbt.dev/>`_, a back-testing library
well-suited for high frequency applications, `PyPortfolioOpt <https://pyportfolioopt.readthedocs.io/en/latest/>`_,
a simple yet powerful library for portfolio optimization that uses well-known models,
`YFinance <https://github.com/ranaroussi/yfinance>`_, which is not a portfolio
optimization library (it only provides a data interface to Yahoo Finance), but
used to be one of our dependencies, and also `CVXPY <https://www.cvxpy.org>`_ by
itself, which is used by some of the above and has an extensive 
`set of examples <https://www.cvxpy.org/examples/index.html#finance>`_
devoted to portfolio optimization (indeed, Cvxportfolio was born out of those).

.. Contributions

Contributions
-------------

We welcome contributors and you don't need to sign a CLA. If you don't have
a Github account you may also send a 
`git patch via email <https://git-scm.com/docs/git-send-email>`_ to the 
`project maintainer <https://github.com/enzbus>`_.

Bug fixes, improvements in the documentations and examples,
new constraints, new cost objects, ..., are good contributions and can be done
even if you're not familiar with the low-level details on the library.
For more advanced contributions we recommend reading the
`TODOs and roadmap
<https://github.com/cvxgrp/cvxportfolio/blob/master/TODOs_ROADMAP.rst>`_
file.

Development
-----------

To set up a development environment locally you should clone the
repository (or, `fork on
Github <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`__
and then clone your fork)

.. code:: bash

   git clone https://github.com/cvxgrp/cvxportfolio.git
   cd cvxportfolio

Then, you should have a look at our
`Makefile <https://www.gnu.org/software/make/manual/make.html#Introduction>`__
and possibly change the ``PYTHON`` variable to match your system’s
python interpreter. Once you have done that,

.. code:: bash

   make env
   make test

This will replicate our `development
environment <https://docs.python.org/3/library/venv.html>`__ and run our
test suite.

You activate the shell environment with one of scripts in ``env/bin``
(or ``env\Scripts`` on Windows), for example if you use bash on POSIX

.. code:: bash

   source env/bin/activate

and from the environment you can run any of the scripts in the examples
(the cvxportfolio package is installed in `editable
mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`__).
Or, if you don't want to activate the environment, you can just run
scripts directly using ``env/bin/python`` (or ``env\Scripts\python`` on
Windows) like we do in the Makefile.

Additionally, to match our CI/CD pipeline, you may set the following
`git hooks <https://git-scm.com/docs/githooks>`__

.. code:: bash

   echo "make lint" > .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   echo "make test" > .git/hooks/pre-push
   chmod +x .git/hooks/pre-push


Code style and quality
----------------------

Cvxportfolio follows the `PEP8 <https://peps.python.org/pep-0008/>`_
specification for code style. This is enforced by the `Pylint
<https://pylint.readthedocs.io/en/stable/>`_ automated linter, with options 
in the `Pyproject 
<https://github.com/cvxgrp/cvxportfolio/blob/master/pyproject.toml>`_
configuration file.
Pylint is also used to enforce code quality standards, along with some of its
optional plugins.
Docstrings are written in the `Sphinx style 
<https://www.sphinx-doc.org/en/master/index.html>`_, are also checked by 
Pylint, and are used to generate the documentation.

.. Versions

Versions and releases
---------------------

Cvxportfolio follows the `semantic versioning <https://semver.org>`_
specification. No breaking change in its public API will be introduced
until the next major version (``2.0.0``), which won't happen for some time. 
New features in the public API are introduced with minor versions 
(``1.1.0``, ``1.2.0``, ...), and only bug fixes at each revision.

The history of our releases (source distributions and wheels) is visible on our 
`PyPI page <https://pypi.org/project/cvxportfolio/#history>`_.

Releases are also tagged in our git repository and include a short summary
of changes in 
`their commit messages <https://github.com/cvxgrp/cvxportfolio/tags>`_.

We maintain a `document listing the planned changes and target releases
<https://github.com/cvxgrp/cvxportfolio/blob/master/TODOs_ROADMAP.rst>`_. 


.. Citing

Citing
------------

If you use Cvxportfolio in work that leads to publication, you can cite the following:

.. code-block:: bibtex

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


The latter is also the first chapter of this PhD thesis:

.. code-block:: bibtex

    @phdthesis{busseti2018portfolio,
        author    = "Busseti, Enzo",
        title     = "Portfolio Management and Optimal Execution via Convex Optimization",
        school    = "Stanford University",
        address   = "Stanford, California, USA",
        month    = "May",
        year     = "2018",
        url     = {\url{https://stacks.stanford.edu/file/druid:wm743bj5020/thesis-augmented.pdf}},
    }


Licensing
---------

Cvxportfolio is licensed under the `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ permissive
open source license.

.. |CVXportfolio on PyPI| image:: https://img.shields.io/pypi/v/cvxportfolio.svg
   :target: https://pypi.org/project/cvxportfolio/
.. |linting: pylint| image:: https://img.shields.io/badge/linting-pylint-yellowgreen
   :target: https://github.com/pylint-dev/pylint
.. |Coverage Status| image:: https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master
   :target: https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/cvxportfolio/badge/?version=stable
   :target: https://cvxportfolio.readthedocs.io/en/stable/?badge=stable
.. |Apache 2.0 License| image:: https://img.shields.io/badge/License-Apache%202.0-green.svg
   :target: https://github.com/cvxgrp/cvxportfolio/blob/master/LICENSE
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/cvxportfolio/badges/version.svg
   :target: https://anaconda.org/conda-forge/cvxportfolio
