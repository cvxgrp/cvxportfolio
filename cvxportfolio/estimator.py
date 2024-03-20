# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements Estimator base classes.

Policies, costs, and constraints inherit from this.
"""

import numbers

import cvxpy as cp
import numpy as np
import pandas as pd

from .errors import DataError, MissingAssetsError, MissingTimesError, NaNError
from .utils import make_numeric, repr_numpy_pandas


class Estimator:
    """Estimator abstraction, designed for repeated evaluation over time.

    Policies, costs, and constraints inherit from this. When overloading
    methods defined here one should be very careful. The recommended usage
    (if you want a class that uses our recursive execution model) is to
    put any sub-estimators at the class attribute level, like we do
    throughout the library. That ensures that the sub-estimators will
    be evaluated before the class itself by both
    :meth:`initialize_estimator_recursive` and
    :meth:`values_in_time_recursive`.
    """

    # explicitely list subestimators, only needed for those not defined
    # at class attribute level
    __subestimators__ = ()

    def initialize_estimator(self, universe, trading_calendar, **kwargs):
        """Initialize estimator instance with universe and trading times.

        This method is called at the start of an online execution, or, in a
        back-test, at its start and whenever the trading universe changes.
        It provides the instance with the current trading universe and a
        :class:`pandas.DatetimeIndex` representing the current and future
        trading calendar, *i.e.*, the times at which the estimator will be
        evaluated, or a best guess of it. The instance uses these to
        appropriately initialize any internal object, such as Cvxpy parameters,
        to the right size (as implied by the universe). Also, especially for
        multi-period optimization and similar policies, awareness of the future
        trading calendar is essential to, *e.g.*, plan in advance.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param trading_calendar: Future (including current) trading calendar.
        :type trading_calendar: pandas.DatetimeIndex
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict
        """

        # we don't raise NotImplementedError because this is called
        # on classes that don't re-define it

    def initialize_estimator_recursive(self, **kwargs):
        """Recursively initialize all estimators in a policy.

        :param kwargs: Parameters sent down an estimator tree to inizialize it.
        :type kwargs: dict
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "initialize_estimator_recursive"):
                subestimator.initialize_estimator_recursive(**kwargs)
        for subestimator in self.__subestimators__:
            subestimator.initialize_estimator_recursive(**kwargs)
        if hasattr(self, "initialize_estimator"):
            self.initialize_estimator(**kwargs)

    def finalize_estimator(self, **kwargs):
        """Finalize estimator instance (currently unused).

        This method is called at the end of an online execution, or, in a
        back-test, whenever the trading universe changes (before calling
        :meth:`initialize_estimator` with the new universe) and at its end.
        We aren't currently using in the rest of the library but we plan to
        move the caching logic in it.

        .. versionadded:: 1.1.0

        :param kwargs: Reserved for future expansion.
        :type kwargs: dict
        """

        # we don't raise NotImplementedError because this is called
        # on classes that don't re-define it

    def finalize_estimator_recursive(self, **kwargs):
        """Recursively finalize all estimators in a policy.

        .. versionadded:: 1.1.0

        :param kwargs: Parameters sent down an estimator tree to finalize it.
        :type kwargs: dict
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "finalize_estimator_recursive"):
                subestimator.finalize_estimator_recursive(**kwargs)
        for subestimator in self.__subestimators__:
            subestimator.finalize_estimator_recursive(**kwargs)
        if hasattr(self, "finalize_estimator"):
            self.finalize_estimator(**kwargs)

    _current_value = None

    @property
    def current_value(self):
        """Current value of this instance.

        :returns: Current value, which can be any object.
        :rtype: numpy.array, pandas.Series, pandas.DataFrame, ...
        """
        return self._current_value

    # pylint: disable=too-many-arguments
    def values_in_time(
            self, t, current_weights, current_portfolio_value,
            past_returns, past_volumes, current_prices,
            mpo_step=None, cache=None, **kwargs):
        """Evaluate estimator at current time, possibly return current value.

        This method is usually the most important for Estimator classes.
        It is called at each point in a back-test with all data of the current
        state. Sub-estimators are evaluated first, in a depth-first recursive
        tree fashion (defined in :meth:`values_in_time_recursive`). The
        signature differs slightly between different estimators, see below.

        :param t: Current timestamp.
        :type t: pandas.Timestamp
        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param current_portfolio_value: Current total value of the portfolio
            in cash units.
        :type current_portfolio_value: float
        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param past_volumes: Past market volumes, or None if not available.
        :type past_volumes: pandas.DataFrame or None
        :param current_prices: Current (open) prices, or None if not available.
        :type current_prices: pandas.Series or None
        :param mpo_step: For :class:`cvxportfolio.MultiPeriodOptimization`
            which step in future planning this estimator is at: 0 is for
            the current step (:class:`cvxportfolio.SinglePeriodOptimization`),
            1 is for day ahead, .... Defaults to ``None`` if unused.
        :type mpo_step: int, optional
        :param cache: Cache or workspace shared between all elements of an
            estimator tree, currently only used by
            :class:`cvxportfolio.MultiPeriodOptimization` (and derived
            classes). It's useful to avoid re-computing expensive things like
            covariance estimates at different MPO steps. Defaults to ``None``
            if unused.
        :type cache: dict, optional
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :returns: Current value of the estimator.
        :rtype: object or None
        """
        # we don't raise NotImplementedError because this is called
        # on classes that don't re-define it

    def values_in_time_recursive(self, **kwargs):
        """Evaluate recursively on sub-estimators.

        :param kwargs: All parameters to :meth:`values_in_time` that are passed
            to all elements contained in a policy object.
        :type kwargs: dict

        :returns: The current value evaluated by this instance, if it
            implements the :meth:`values_in_time` method and it returns
            something there.
        :rtype: numpy.array, pandas.Series, pandas.DataFrame, ...
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "values_in_time_recursive"):
                subestimator.values_in_time_recursive(**kwargs)
        for subestimator in self.__subestimators__:
            subestimator.values_in_time_recursive(**kwargs)
        if hasattr(self, "values_in_time"):
            # pylint: disable=assignment-from-no-return
            self._current_value = self.values_in_time(**kwargs)
            return self.current_value
        return None

    def collect_hyperparameters(self):
        """Collect (recursively) all hyperparameters defined in a policy.

        :returns: List of :class:`cvxportfolio.hyperparameters.HyperParameter`
            instances.
        :rtype: list
        """
        result = []
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "collect_hyperparameters"):
                result += subestimator.collect_hyperparameters()
        for subestimator in self.__subestimators__:
            result += subestimator.collect_hyperparameters()

        # TODO: here list(set(result)) would take care of duplicate references,
        # but current logic of optimize_hyperparameters would break.
        # Current approach is correct logically, but may run duplicate
        # bts in optimize_hyperparameters if there are duplicate refs
        return result

    def __repr__(self):
        """Pretty-print the cvxportfolio object in question.

        We make sure that every object the user interacts with can be
        pretty-printed to the interpreter, ideally in a way such that
        copy-pasting the output to the prompt results in an identical object.
        We can't do that all the times but we do our best. This is used
        throughout the library, for example it is included in backtest results
        so the user knows which policy generated that backtest, .... We prefer
        to define the logic of this directly insted of relying, e.g., on
        dataclasses logic, because we want to customize it to our usecase.
        """
        lhs = self.__class__.__name__ + '('
        core = ''
        for name, attr in self.__dict__.items():
            if attr is None:
                continue
            if hasattr(attr, "values_in_time_recursive") or\
                    hasattr(attr, "values_in_time") or (name[0] != '_'):
                core += name + '=' + attr.__repr__() + ', '
        core = core[:-2]  # remove trailing comma and space if present
        rhs = ')'
        return lhs + core + rhs


class SimulatorEstimator(Estimator):
    """Base class for estimators that are used by the market simulator.

    .. versionadded:: 1.2.0

    This is currently used as the base class for
    :class:`cvxportfolio.costs.SimulatorCost` but could implement in future
    versions more operations done as part of a simulation (back-test) loop,
    like filtering out trades (rejecting small ones, ...), or more. It allows
    for nested evaluation like it's done by :meth:`values_in_time` and
    :meth:`values_in_time_recursive`. Estimators that are used in the market
    simulator should derive from this. Some examples are
    :class:`DataEstimator`, which is used to select values (like borrow costs)
    in simulations as well as in optimization, and
    :class:`cvxportfolio.forecast.HistoricalStandardDeviation`, which is used
    also in simulation by :class:`cvxportfolio.costs.TransactionCost`.
    """

    def simulate( # pylint: disable=too-many-arguments
        self, t, t_next, u, h_plus, past_volumes,
        past_returns, current_prices, current_returns, current_volumes,
        current_weights, current_portfolio_value, **kwargs):
        """Evaluate the estimator as part of a Market Simulator back-test loop.

        Cost classes that are meant to be used in the simulator can
        implement this. The arguments to this are the same as for
        :meth:`cvxportfolio.estimator.Estimator.values_in_time` plus the
        realized returns and volumes in the period, and the trades requested
        by the policy, ....

        :param t: Current timestamp.
        :type t: pandas.Timestamp
        :param u: Trade vector in cash units requested by the policy.
            If the market simulator implements rounding by number of shares
            and/or canceling trades on assets whose volume for the period
            is zero, this is after those transformations.
        :type u: pandas.Series
        :param h_plus: Post-trade holdings vector.
        :type h_plus: pandas.Series
        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param current_returns: Current period's market returns (including
            cash).
        :type current_returns: pandas.Series
        :param past_volumes: Past market volumes, or None if not available.
        :type past_volumes: pandas.DataFrame or None
        :param current_volumes: Current period's market volumes, or None if not
            available.
        :type current_volumes: pandas.Series or None
        :param current_prices: Current (open) prices, or None if not available.
        :type current_prices: pandas.Series or None
        :param current_weights: Current allocation weights (before trading).
        :type current_weights: pandas.Series
        :param current_portfolio_value: Current total value of the portfolio
            in cash units, before costs.
        :type current_portfolio_value: float
        :param t_next: Timestamp of the next trading period.
        :type t_next: pandas.Timestamp
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :returns: The current value of this instance.
        :rtype: any object
        """
        raise NotImplementedError # pragma: no cover

    def simulate_recursive(self, **kwargs):
        """Evaluate simulator value(s) recursively on sub-estimators.

        :param kwargs: All parameters to :meth:`simulate` that are passed
            to all elements contained in a simulator cost object.
        :type kwargs: dict

        :returns: The current simulator value evaluated by this instance, if it
            implements the :meth:`simulate` method and it returns
            something there.
        :rtype: numpy.array, pandas.Series, pandas.DataFrame, ...
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "simulate_recursive"):
                subestimator.simulate_recursive(**kwargs)
        if hasattr(self, "simulate"):
            self._current_value = self.simulate(**kwargs)
            return self.current_value
        return None # pragma: no cover

class CvxpyExpressionEstimator(Estimator):
    """Base class for estimators that are Cvxpy expressions."""

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile term to cvxpy expression.

        This is called by a Policy class on its terms before the start
        of the backtest to compile its Cvxpy problem. If the Policy
        changes in time this is called at every time step.

        It can either return a scalar expression, in the case of
        objective terms, or a list of cvxpy constraints, in the case of
        constraints.

        In MultiPeriodOptimization policies this is called separately
        for costs and constraints at different look-ahead steps with the
        corresponding w_plus and z.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        """
        raise NotImplementedError # pragma: no cover

# pylint: disable=too-many-arguments
class DataEstimator(SimulatorEstimator):
    """Estimator of point-in-time values from internal data.

    It also implements logic to check that no ``nan`` are returned
    by its ``values_in_time_recursive`` method, which is the way Cvxportfolio
    objects use this class to get data, to compile and update a Cvxpy
    parameter, and to slice the data with the current trading universe.

    :param data: Data expressed preferably as pandas Series or DataFrame
        where the first index is a ``pandas.DateTimeIndex``. Otherwise you can
        pass a callable object which implements the
        :meth:`values_in_time_recursive` method (with the standard signature)
        and returns the corresponding value in time, or a constant float,
        numpy.array, or even pandas Series or DataFrame not indexed by time
        (e.g., a covariance matrix where both index and columns are the stock
        symbols).
    :type data: object, pandas.Series, pandas.DataFrame
    :param use_last_available_time: if the pandas index exists
        and is a ``pandas.DateTimeIndex`` you can instruct
        :meth:`values_in_time_recursive` to retrieve the last available value
        at time t by setting this to True. Default is False.
    :type use_last_available_time: bool
    :param allow_nans: If True, allow data returned to contain ``nan``.
        Default False.
    :type allow_nans: bool
    :param compile_parameter: If True, compile a Cvxpy parameter that gets
        updated with the current value of the instance at each point in a
        backtest. Default False.
    :type compile_parameter: bool
    :param non_negative: If True, the compiled Cvxpy parameter is non-negative
        (this affects certain Cvxpy operations). Default False.
    :type non_negative: bool
    :param positive_semi_definite: If True, the compiled Cvxpy parameter is
        market as a positive semi-definite matrix (this affects certain Cvxpy
        operations). Default False.
    :type positive_semi_definite: bool
    :param data_includes_cash: If True, when the data is sliced with the
        current trading universe we also look for the values corresponding to
        the cash account. Default False.
    :type data_includes_cash: bool
    :param ignore_shape_check: If True, we don't do any slicing of the data
        according to the current trading universe. Default False.
    :type ignore_shape_check: bool

    :raises cvxportfolio.NaNError: If ``nan`` are present in result.
    :raises cvxportfolio.MissingTimesError: If some times are missing.
    :raises cvxportfolio.MissingAssetsError: If some assets are missing.
    :raises cvxportfolio.DataError: If data is not in the right form.
    """

    def __init__(
            self, data, use_last_available_time=False, allow_nans=False,
            compile_parameter=False, non_negative=False,
            positive_semi_definite=False, data_includes_cash=False,
            ignore_shape_check=False):
        self.data = make_numeric(data)
        self._use_last_available_time = use_last_available_time
        self._allow_nans = allow_nans
        self._compile_parameter = compile_parameter
        self._non_negative = non_negative
        self._positive_semi_definite = positive_semi_definite
        self._universe_maybe_noncash = None
        self._data_includes_cash = data_includes_cash
        self._ignore_shape_check = ignore_shape_check
        self.parameter = None

    def initialize_estimator(self, universe, trading_calendar, **kwargs):
        """Initialize with current universe.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param trading_calendar: Future (including current) trading calendar.
        :type trading_calendar: pandas.DatetimeIndex
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """

        self._universe_maybe_noncash = \
            universe if self._data_includes_cash else universe[:-1]

        if self._compile_parameter:
            # to make sure it doesn't try to update the parameter
            self.parameter = None
            value = self.values_in_time_recursive(
                t=trading_calendar[0])
            self.parameter = cp.Parameter(
                value.shape if hasattr(value, "shape") else (),
                PSD=self._positive_semi_definite, nonneg=self._non_negative)

    def value_checker(self, result):
        """Ensure that only scalars or arrays without np.nan are returned.

        :raises cvxportfolio.errors.NaNError: If NaNs are found.
        :raises cvxportfolio.errors.DataError: If the value passed is not a
            scalar or Numpy array.

        :param result: Scalar or array that we check has no NaNs.
        :type result: float or numpy.array

        :returns: Input value; if array, a copy.
        :rtype: float or numpy.array
        """

        if isinstance(result, numbers.Number):
            if np.isnan(result) and not self._allow_nans:
                raise NaNError(
                    f"{self.__class__.__name__}.values_in_time_recursive"
                    + " result is a np.nan scalar.")
            return result

        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)) and not self._allow_nans:
                message = f"{self.__class__.__name__}.values_in_time_recursive"
                message += " result is an array with np.nan's."
                raise NaNError(message)
            # we pass a copy because it can be accidentally overwritten
            return np.array(result)

        raise DataError(
            f"{self.__class__.__name__}.values_in_time_recursive result"
            + " is not a scalar or array.")

    def _universe_subselect(self, data):
        """This function subselects from ``data`` the relevant universe.

        See github issue #106.

        If data is a pandas Series we subselect its index. If we fail we
        throw an error. If data is a pandas DataFrame (covariance,
        exposure matrix) we try to subselect its index and columns. If
        we fail on either we ignore the failure, but if we fail on both
        we throw an error. If data is a numpy 1-d array we check that
        its length is the same as the universe's. If it is a 2-d array
        we check that at least one dimension is the same as the
        universe's. If the universe is None we skip all checks. (We may
        revisit this choice.) This only happens if the DataEstimator
        instance is not part of a Estimator tree (a usecase which we
        will probably drop).
        """

        if (self._universe_maybe_noncash is None) or self._ignore_shape_check:
            return data.values if hasattr(data, 'values') else data

        if isinstance(data, pd.Series):
            try:
                return data.loc[self._universe_maybe_noncash].values
            except KeyError as exc:
                raise MissingAssetsError(
                    f"The pandas Series found by {self.__class__.__name__} has"
                    + f" index {data.index} while the current universe"
                    + (' minus cash ' if not self._data_includes_cash else ' ')
                    + f"is {self._universe_maybe_noncash}; It was not possible"
                    + " to reconcile the two.") from exc

        if isinstance(data, pd.DataFrame):
            try:
                return data.loc[self._universe_maybe_noncash,
                    self._universe_maybe_noncash].values
            except KeyError:
                try:
                    return data.loc[:, self._universe_maybe_noncash].values
                except KeyError:
                    try:
                        return data.loc[self._universe_maybe_noncash, :].values
                    except KeyError:
                        pass
            raise MissingAssetsError(
                f"The pandas DataFrame found by {self.__class__.__name__} has"
                + f" index {data.index} and columns {data.columns} while the "
                + "current universe"
                + (' minus cash ' if not self._data_includes_cash else ' ')
                + f"is {self._universe_maybe_noncash}; It was not possible"
                + " to reconcile the two.")

        if isinstance(data, np.ndarray):
            dimensions = data.shape
            if not len(self._universe_maybe_noncash) in dimensions:

                raise MissingAssetsError(
                    f"The numpy array found by {self.__class__.__name__}"
                    + f" has dimensions {data.shape}"
                    + " while the current universe"
                + f"{' minus cash ' if not self._data_includes_cash else ' '}"
                    + f"has size {len(self._universe_maybe_noncash)};"
                    + " It was not possible to reconcile the two.")
            return data

        # scalar
        return data

    def _internal_values_in_time(self, t, **kwargs):
        """Internal method called by :meth:`values_in_time`."""

        # here we trust the result (change?)
        if hasattr(self.data, "values_in_time_recursive"):
            return self.data.current_value

        # here (probably user-provided) we check
        if hasattr(self.data, "values_in_time"):
            if len(kwargs) == 0:
                raise ValueError(
                    "It seems you're using a custom forecaster as part of a "
                    "simulate_recursive evaluation, you should derive from "
                    "SimulatorEstimator instead.")
            return self.value_checker(self._universe_subselect(
                self.data.current_value if hasattr(self.data, 'current_value')
                else self.data.values_in_time(t=t, **kwargs)))

        # if self.data is pandas and has datetime (first) index
        if (hasattr(self.data, "loc") and hasattr(self.data, "index")
            and (isinstance(self.data.index, pd.DatetimeIndex)
                 or (isinstance(self.data.index, pd.MultiIndex) and
                     isinstance(self.data.index.levels[0],
                         pd.DatetimeIndex)))):
            try:
                if self._use_last_available_time:
                    if isinstance(self.data.index, pd.MultiIndex):
                        newt = self.data.index.levels[0][
                            self.data.index.levels[0] <= t][-1]
                    else:
                        newt = self.data.index[self.data.index <= t][-1]
                    tmp = self.data.loc[newt]
                else:
                    tmp = self.data.loc[t]

                return self.value_checker(self._universe_subselect(tmp))

            except (KeyError, IndexError) as exc:
                raise MissingTimesError(
                    f"{self.__class__.__name__} could not find data"
                    + f" for time {t}. The datetime index provided is: "
                    + (str(self.data.index.levels[0]) if hasattr(
                        self.data.index, 'levels') else str(self.data.index))
                    + ". This could be due to wrong timezone"
                    + " setting: in general Cvxportfolio objects are timezone"
                    + "-aware, the data you pass should be as well.") from exc

        # if data is pandas but no datetime index (constant in time)
        if hasattr(self.data, "values"):
            return self.value_checker(self._universe_subselect(self.data))

        # if data is scalar or numpy
        return self.value_checker(self._universe_subselect(self.data))

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Obtain value of `self.data` at time t or right before.

        :param kwargs: All parameters passed to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.NaNError: The data provided contains
            NaNs at current time.

        :returns: The  value from this
            :class:`cvxportfolio.estimator.DataEstimator` at current time.
        :rtype: int, float, numpy.ndarray
        """
        try:
            result = self._internal_values_in_time(**kwargs)
        except NaNError as exc:
            raise NaNError(f"{self.__class__.__name__} found NaNs"
                + f" at time {kwargs['t']}.") from exc
        if self.parameter is not None:
            self.parameter.value = result
        return result

    def simulate( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Evaluate in simulation (e.g., TransactionCost).

        :param t: Current timestamp.
        :type t: pd.Timestamp
        :param kwargs: All other unused arguments to
            :meth:`SimulatorEstimator.simulate`.
        :type kwargs: dict

        :returns: The  value from this
            :class:`cvxportfolio.estimator.DataEstimator` at current time.
        :rtype: int, float, numpy.ndarray
        """
        # We don't support evaluation inside DataEstimator in this case
        # You should implement simulate/simulate_recursive in each
        # SimulatorEstimator wrapped by DataEstimator
        return self.values_in_time(t=t)

    def __repr__(self):
        """Pretty-print."""
        if np.isscalar(self.data):
            return str(self.data)
        if hasattr(self.data, 'values_in_time_recursive'
            ) or hasattr(self.data, 'values_in_time'):
            return self.data.__repr__()
        return repr_numpy_pandas(self.data)
