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
  ``_loader_*`` and ``_storer_*`` from ``cvxportfolio.data``. 

``cvxportfolio.forecast``
-------------------------

- cache logic needs improvement, not easily exposable to third-parties now with ``dataclass.__hash__``
- [ ] drop decorator
- [ ] drop dataclass, PR #133
- [ ] cache IO logic should be managed by forecaster not by simulator, could be done by ``initialize_estimator``; maybe enough to just
  define it in the base class of forecasters
- [ ] lots of stuff can be re-used at universe change
- [X] generalize the mean estimator
- [X] use same code for ``past_returns``, ``past_returns**2``, ``past_volumes``, .... Done in #126, target ``1.2.0``
- [X] add rolling window option, should be in ``pd.Timedelta``. Done in #126, target ``1.2.0``
- [X] add exponential moving avg, should be in half-life ``pd.Timedelta``. Done in #126, target ``1.2.0``
- [X] add same extras to the covariance estimator. Done in #126, target ``1.2.0``
- goal: make this module crystal clear; third-party ML models should use it (at least for caching)

``cvxportfolio.estimator``
--------------------------

- [ ] ``DataEstimator`` needs refactoring, too long and complex methods.
- [ ] ``Estimator`` could define base logic for on-disk caching. By itself it
  wouldn't do anything, actual functionality implemented by forecasters' base
  class.

- [ ] ``initialize_estimator`` could get optional market data partial
    signature for caching. Default None, no incompatible change.
- [X] Could get a ``finalize_estimator`` method used for storing
    data, like risk models on disk, doesn't need arguments; it can use the
    partial signature got above. No incompatible change.
- [ ] Leverage ``finalize_estimator``; obvious one is in ``CvxpyExpressionEstimator``,
  delete all cvxpy parameters. Need to be carefully tested for memory safety
  (cvxpy semi-compiled objects are **not** memory safe). Same for the parameter
  in ``DataEstimator``.
- [ ] To Estimators throughout; set internal variables to None as necessary.

``cvxportfolio.data``
--------------------------

- [X] Handle user-defined, time-varying investable universes both in `UserProvidedMarketData`
  and `DownloadedMarketData`. Requested in #137. Idea: add `investable_assets_at_times` parameter to
  both. It is specified as ``bool`` dataframe with datetime index and as columns all assets in the
  universe. From the datetime index it is selected at each point in time the most recent line, ``DataEstimator``
  already has that capability. Make sure logging throughout the library is accurate about changes in investable universe.
  The selection is done in addition to the ``min_history`` filtering and non-``nan`` returns for the period.
  (We don't want to lose the guarantees coming from those.)
- [X] Improve ``YahooFinance`` data cleaning. Idea is to factor it in a 
  base ``OpenLowHighCloseVolume`` class, which should flexible enough to
  accommodate adjusted closes (i.e., with backwards dividend adjustments like
  YF), total returns like other data sources, or neither for non-stocks assets.
  This would implement all data cleaning process as sequence of small steps
  in separate methods, with good logging. It would also implement data quality
  check in the ``preload`` method to give feedback to the user. PR #127
- [X] Factor ``data.py`` in ``data/`` submodule. PR #127
- [ ] Consider factoring cleaning methods for use by `UserProvidedMarketData` as
  well; many choices would need to be done. Has been requested in #137 to work
  with user-provided prices and volumes in shares, doing internal conversion and
  cleaning.

``cvxportfolio.simulator``
--------------------------
- [ ] Make ``BackTestResult`` interface methods with ``MarketSimulator`` 
  public. It probably should do a context manager b/c logging code in 
  ``BackTestResult`` does cleanup of loggers at the end, to ensure all right
  in case back-test fails, or ends in bankruptcy, ...
- [ ] Move caching logic out of it; see above.
- [ ] Make sure not touching any method specific to Cvxpy policies.
- [ ] Make clear where to subclass for changing simulate logic (which seems
  already understandable).
- [ ] Simplify a lot backtest logic flow; ``backtest_many`` should parallelize ``backtest``;
  ``backtest`` should incorporate the logic of ``_backtest``.
- [ ] Remove all ``copy.deepcopy``. Depends on robustifying testing, tricky
  issues are: re-use of Policy objects, multiprocessing, parametric semi-compiled
  Cvxpy object (**not** memory safe). Testing should do many intersections of
  these. Hunch that ``finalize_estimator`` is key.

``cvxportfolio.risks``
----------------------
- [ ] Consider adding user-specified benchmark (policy) object that supersedes
  the one defined by the S/MPO policy. To do it clean needs some base methods
  and ``super()`` invocations, not sure if worth it.

``cvxportfolio.hyperparameters``
-------------------------
Partially public; only ``cvx.Gamma()`` (no arguments) and ``optimize_hyperparameters``
(simple usage) are public, all the rest is not.

- [ ] Clean up interface w/ ``MarketSimulator``, right now it calls private 
  methods, maybe enough to make them public.
- [ ] Add risk/fine default ``GammaTrade``, ``GammaRisk`` (which are
  ``RangeHyperParameter``) modeled after original examples from paper.
- [X] Add ``Constant`` internal object throughout the library, also in ``DataEstimator``
  in the case of scalar; it resolves to ``current_value`` if you pass a hyper-parameter.
    Replaced with _resolve_hyperpar in #126.
- [ ] Distinguish integer and positive hyper-parameters (also enforced by Constant).
- [ ] Consider changing the increment/decrement model; hyperparameter object
  could instead return a ``neighbors`` set at each point. Probably cleaner.
- [ ] Together with rationalization of magic methods of ``Cost`` (removal of 
  ``CombinedCost``, ...), magic methods should be rationalized here, and similar
  logic should be used; cleaner and simpler. Hyperparameters are (mostly),
  symbolic scalars, but can also be timedeltas, ... Will probably have to the thought
  out a bit.

``cvxportfolio.policies``
-------------------------

- [ ] Add `AllIn` policy, which allocates all to a single name (like 
  ``AllCash``).

Optimization policies
~~~~~~~~~~~~~~~~~~~~~

- [ ] Improve behavior for infeasibility/unboundedness/solver error. Idea:
  optimization policy gets arguments ``infeasible_fallback``, ... which are
  policies (default to ``cvx.Hold``), problem is that this breaks
  compatibility, it doesn't if we don't give defaults (so exceptions are raised
  all the way to the caller), but then it's extra complication (more 
  arguments). Consider for ``2.0.0``.
- [X] Improve ``__repr__`` method, now hard to read. Target ``1.1.0``.
- [ ] Leverage ``finalize_estimator`` to delete cvxpy problem object. Needs to
  be carefully tested (see note in simulator). Order of deletion of problem
  and parameters might matter. Cvxpy semi-compiled objects are **not** memory safe;
  We're currently defaulting to Python's garbage collection by using ``copy.deepcopy``
  at the source.
- [ ] Rationalize usage of ``is_dcp``, ``is_convex``, ``is_concave``, ``is_dpp``,
  some of those were put there as sanity checks, some are used to simplify error
  resolution and give to the user a pointer to their wrong syntax, some are used
  as guard against misspecified custom terms, ... This applies throughout the
  optimization-based terms.

``cvxportfolio.costs``
----------------------
- [ ] ``CombinedCost`` to be removed. Cleaner to do ``SumCosts``
  and ``MulCosts`` with good logic in the magic methods of ``Cost`` and simple
  recursive (using Python's arithmetics) resolution. Also can get rid of overridden
  base methods. ``SumCosts`` has left and right children (can only sum costs!),
  ``MulCosts`` has cost and scalar/hyperpar (can not multiply costs!).
  Sub and div are handled by same,
  so than we can put in the examples ``cvx.FullCovariance() / 2.``!
- [ ] Rethink logic that does convexity check in ``CombinedCost`` (if we even want it there).

``cvxportfolio.constraints``
----------------------------

- [ ] Add missing constraints from the paper. List of currently missing ones follows.
- [ ] Limits relative to asset capitalization, page 34; need to make sure interface
  is sensible.
- [ ] No-hold constraint; is there in some form (depends on trading period).
  Needs cleaning.
- [ ] Stress constraints, page 35. Nice one, need to make sure interface
  is sensible.
- [ ] Liquidation loss constraint, page 36. Should be feasible now that ``TransactionCost``
  interface has been finalized.
- [ ] Concentration limit, page 36. Can be done easily but it's inefficient and
  not very useful.
- [ ] Limits relative to trading volume, page 37. Should be easy now that ``HistoricalMeanVolume``
  has been formalized.
- [ ] No buy/sell/trade, page 37. Again it's already present in some form (e.g. ``MaxTrade`` with
  time-changing limit) but needs to be clarified/formalized.
- [X] Make ``MarketNeutral`` accept arbitrary benchmark (policy object). Done in #126.

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

- [ ] Add extra pylint checkers: code complexity.
- [ ] Consider removing downloaded data from ``test_simulator.py``,
  so only ``test_data.py`` requires internet. Work in progress PR #140

Documentation
-------------

- [ ] Add plots and words to more examples (ideally all that are exposed in HTML);
  current code in Makefile for that is not good; make a script so you can run
  each example by itself.
- [X] Manual. PR #124
- [X] Quickstart, probably to merge into manual. PR #124

Examples
--------

- [ ] Finish restore examples from paper. Work in progress PR #143
- [ ] Consider making examples a package that can be pip installed.
- [ ] If not pip-installable, clarify as well as possible that many examples
  need to be run with ``python -m examples. ...`` because they import shared
  resources.
