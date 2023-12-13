# Features & refactoring

### `cache.py`

- probably to be dropped; should use `_loader_*` and `_storer_*` from `data.py`

### `forecast.py`

- cache logic needs improvement, not easily exposable to third-parties now with `dataclass.__hash__`
  - drop decorator
  - drop dataclass
  - cache IO logic should be managed by forecaster not by simulator, could be done by `initialize_estimator`; maybe enough to just
    define it in the base class of forecasters
- improve names of internal methods, clean them (lots of stuff can be re-used at universe change, ...)
- generalize the mean estimator:
  - use same code for `past_returns`, `past_returns**2`, `past_volumes`, ...
  - add rolling window option, should be in `pd.Timedelta`
  - add exponential moving avg, should be in half-life `pd.Timedelta`
- add same extras to the covariance estimator
- goal: make this module crystal clear; third-party ML models should use it (at least for caching)

### `estimator.py`

- `DataEstimator` needs refactoring, too long and complex methods



### Development & testing
- add extra pylint checkers: 
  - code complexity
- consider removing downloaded data from `test_simulator.py`, so only `test_data.py` requires internet 

## Documentation

## Examples
