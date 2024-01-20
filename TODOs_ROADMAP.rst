TODOs and ROADMAP
=================

Cvxportfolio follows the `semantic versioning <https://semver.org>`_
specification for changes in its public API. The public API is defined
as:

- public methods, *i.e.*, without a leading underscore ``_``
- methods and objects clearly documented as such and/or used in the examples.

*Internal* methods that are used by 
Cvxportfolio objects to communicate with each other (or other tasks) and don't
have a leading underscore, are considered public if they are exposed through 
the HTML documentation and/or are used in the examples.

In this document we list the planned
changes, in particular the ones that are relevant for semantic versioning, and 
their planned release.

``cvxportfolio.cache``
----------------------

- [ ] Not part of public API; to be redesigned and probably dropped. Should use
  ``_loader_*`` and ``_storer_*`` from ``cvxportfolio.data``. Target ``1.1.0``.

``cvxportfolio.forecast``
-------------------------

- cache logic needs improvement, not easily exposable to third-parties now with ``dataclass.__hash__``

  - drop decorator
  - drop dataclass
  - cache IO logic should be managed by forecaster not by simulator, could be done by ``initialize_estimator``; maybe enough to just
    define it in the base class of forecasters
- improve names of internal methods, clean them (lots of stuff can be re-used at universe change, ...)
- generalize the mean estimator:

  - use same code for ``past_returns``, ``past_returns**2``, ``past_volumes``, ...
  - add rolling window option, should be in ``pd.Timedelta``
  - add exponential moving avg, should be in half-life ``pd.Timedelta``
- add same extras to the covariance estimator
- goal: make this module crystal clear; third-party ML models should use it (at least for caching)

``cvxportfolio.estimator``
--------------------------

- [ ] ``DataEstimator`` needs refactoring, too long and complex methods. Target 
  ``1.1.1``. 
- ``Estimator`` could define base logic for on-disk caching. By itself it
  wouldn't do anything, actual functionality implemented by forecasters' base
  class.

  - [ ] ``initialize_estimator`` could get optional market data partial
    signature for caching. Default None, no incompatible change.
  - [X] Could get a ``finalize_estimator`` method used for storing
    data, like risk models on disk, doesn't need arguments; it can use the
    partial signature got above. No incompatible change.

``cvxportfolio.data``
--------------------------

- [ ] Improve ``YahooFinance`` data cleaning. Idea is to factor it in a 
  base ``OpenLowHighCloseVolume`` class, which should flexible enough to
  accommodate adjusted closes (i.e., with backwards dividend adjustments like
  YF), total returns like other data sources, or neither for non-stocks assets.
  This would implement all data cleaning process as sequence of small steps
  in separate methods, with good logging. It would also implement data quality
  check in the ``preload`` method to give feedback to the user. PR #125
- [ ] Factor ``data.py`` in ``data/`` submodule. PR #125

``cvxportfolio.simulator``
--------------------------
- [ ] Make ``BackTestResult`` interface methods with ``MarketSimulator`` 
  public. It probably should do a context manager b/c logging code in 
  ``BackTestResult`` does cleanup of loggers at the end, to ensure all right
  in case back-test fails. 
- [ ] Move caching logic out of it; see above.

``cvxportfolio.risks``
----------------------

``cvxportfolio.hyperparameters``
-------------------------
Partially public; only ``cvx.Gamma()`` (no arguments) and ``optimize_hyperparameters``
(simple usage) are public, all the rest is not.

- [ ] Clean up interface w/ ``MarketSimulator``, right now it calls private 
  methods, maybe enough to make them public. Target ``1.1.1``.
- [ ] Add risk/fine default ``GammaTrade``, ``GammaRisk`` (which are
  ``RangeHyperParameter``) modeled after original examples from paper.
- [ ] Add ``Constant`` internal object throughout the library, also in ``DataEstimator``
  in the case of scalar; it resolves to ``current_value`` if you pass a hyper-parameter.
- [ ] Distinguish integer and positive hyper-parameters (also enforced by Constant).
- [ ] Consider changing the increment/decrement model; hyperparameter object
  could instead return a ``neighbors`` set at each point. Probably cleaner.

``cvxportfolio.policies``
-------------------------

- [ ] Add `AllIn` policy, which allocates all to a single name (like 
  ``AllCash``). Target ``1.1.0``.

Optimization policies
~~~~~~~~~~~~~~~~~~~~~

- [ ] Improve behavior for infeasibility/unboundedness/solver error. Idea:
  optimization policy gets arguments ``infeasible_fallback``, ... which are
  policies (default to ``cvx.Hold``), problem is that this breaks
  compatibility, it doesn't if we don't give defaults (so exceptions are raised
  all the way to the caller), but then it's extra complication (more 
  arguments). Consider for ``2.0.0``.
- [X] Improve ``__repr__`` method, now hard to read. Target ``1.1.0``.

``cvxportfolio.constraints``
----------------------------

- [ ] Add missing constraints from the paper.
- [ ] Make ``MarketNeutral`` accept arbitrary benchmark (policy object).

``cvxportfolio.result``
-----------------------

- [ ] Add a ``bankruptcy`` property (boolean). Amend ``sharpe_ratio``
  and other aggregate statistics (as best as possible) to return ``-np.inf``
  if back-test ended in backruptcy. This is needed specifically for
  hyper-parameter optimization. Target ``1.1.1``.
- [X] Capture **logs** from the back-test; add ``logs`` property that returns
  them as a string (newline separated, like a .log file). Make log level
  changeable by a module constant (like ``cvxportfolio.result.LOG_LEVEL``) set
  to ``INFO`` by default. Then, improve logs throughout (informative, proactive
  on possible issues). Logs formatter should produce source module and
  timestamp.

Other 
-----

- [X] Exceptions are not too good, probably ``cvxportfolio.DataError`` should
  be ``ValueError``, .... Research this, one option is to simply derive from
  built-ins (``class DataError(ValueError): pass``), .... No compatibility
  breaks.

Development & testing
---------------------

- [ ] Add extra pylint checkers. 
  
  - [ ] Code complexity. Target ``1.1.1``. 
- [ ] Consider removing downloaded data from ``test_simulator.py``,
  so only ``test_data.py`` requires internet. 

Documentation
-------------

- [ ] Improve examples section, also how "Hello world" is mentioned in readme.
- [ ] Manual. PR #124
- [ ] Quickstart, probably to merge into manual. PR #124

Examples
--------

- [ ] Finish restore examples from paper. Target ``1.1.1``.
- [ ] Expose more (all?) examples through HTML docs.
- [ ] Consider making examples a package that can be pip installed.
