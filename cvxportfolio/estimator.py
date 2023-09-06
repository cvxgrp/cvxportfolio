# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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

"""
This module implements Estimator base classes. Policies, costs, and constraints inherit
from this.
"""

import numpy as np
import pandas as pd
import cvxpy as cp


from .errors import MissingTimesError, DataError, NaNError, MissingAssetsError
from .hyperparameters import HyperParameter
from .utils import repr_numpy_pandas


class Estimator:
    """Estimator base class.

    Policies, costs, and constraints inherit from this. When overloading
    methods defined here one should be careful on deciding whether to call
    the `super()` corresponding method. It can make sense to call it before
    some logic, after, or not calling it at all. Also, any subclass of this that uses
    logic defined here should be careful to put estimator subclasses at the class
    attribute level, so that the two methods defined here get called recursively
    on them.
    """

    def _recursive_values_in_time(self, **kwargs):
        """Evaluates estimator at a point in time recursively on its sub-estimators.

        This function is called by Simulator classes on Policy classes
        returning the current trades list. Policy classes, if
        they contain internal estimators, should declare them as attributes
        and call this base function (via `super()`) before
        they do their internal computation. CvxpyExpression estimators
        should instead define this method to update their Cvxpy parameters.

        Once we finalize the interface all parameters will be listed here.
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "_recursive_values_in_time"):
                subestimator._recursive_values_in_time(**kwargs)
        if hasattr(self, "_values_in_time"):
            self.current_value = self._values_in_time(**kwargs)
            return self.current_value

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
            if hasattr(attr, "_recursive_values_in_time") or \
                    hasattr(attr, "_values_in_time") or (name[0] != '_'):
                core += name + '=' + attr.__repr__() + ', '
        core = core[:-2]  # remove trailing comma and space if present
        rhs = ')'
        return lhs + core + rhs


class PolicyEstimator(Estimator):
    """Base class for (most) estimators that are part of policy objects."""

    def _collect_hyperparameters(self):
        """This method finds all hyperparameters defined as part of a policy.
        """
        result = []
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "_collect_hyperparameters"):
                result += subestimator._collect_hyperparameters()
        return result

    def _recursive_pre_evaluation(self, universe, backtest_times):
        """Recursively initialize estimator tree for backtest.

        :param universe: names of assets to be traded 
        :type universe: pandas.Index
        :param backtest_times: times at which the estimator will be evaluated
        :type backtest_time: pandas.DatetimeIndex
        """
        for _, subestimator in self.__dict__.items():
            if hasattr(subestimator, "_recursive_pre_evaluation"):
                subestimator._recursive_pre_evaluation(
                    universe, backtest_times)
        if hasattr(self, "_pre_evaluation"):
            self._pre_evaluation(universe, backtest_times)


class CvxpyExpressionEstimator(PolicyEstimator):
    """Base class for estimators that are Cvxpy expressions."""

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile term to cvxpy expression.

        This is called by a Policy class on its terms before the start of the backtest
        to compile its Cvxpy problem. If the Policy changes in time
        this is called at every time step.

        It can either return a scalar expression, in the case of objective terms,
        or a list of cvxpy constraints, in the case of constraints.

        In MultiPeriodOptimization policies this is called separately
        for costs and constraints at different look-ahead steps with
        the corresponding w_plus and z.


        :param w_plus: post-trade weights 
        :type w_plus: cvxpy.Variable
        :param z: trade weights 
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: post-trade weights minus benchmark weights 
        :type w_plus_minus_w_bm: cvxpy.Variable
        """
        raise NotImplementedError


class DataEstimator(PolicyEstimator):
    """Estimator of point-in-time values from internal `self.data`.

    It also implements logic to check that no `np.nan` are returned
    by its `_recursive_values_in_time` method, which is the way `cvxportfolio`
    objects use this class to get data.

    :param data: Data expressed preferably as pandas Series or DataFrame 
        where the first index is a pandas.DateTimeIndex. Otherwise you can
        pass a callable object which implements the _recursive_values_in_time method
        (with the standard signature) and returns the corresponding value in time,
        or a constant float, numpy.array, or even pandas Series or DataFrame not
        indexed by time (e.g., a covariance matrix where both index and columns
        are the stock symbols).
    :type data: object, pandas.Series, pandas.DataFrame 
    :param use_last_available_time: if the pandas index exists
        and is a pandas.DateTimeIndex you can instruct self._recursive_values_in_time
        to retrieve the last available value at time t by setting
        this to True. Default is False.
    :type use_last_available_time: bool 
    
    :raises cvxportfolio.NaNError: If np.nan's are present in result.
    :raises cvxportfolio.MissingTimesError: If some times are missing.
    :raises cvxportfolio.MissingAssetsError: If some assets are missing.
    :raises cvxportfolio.DataError: If data is not in the right form.
    """

    def __init__(self, data, use_last_available_time=False, allow_nans=False,
                 compile_parameter=False, non_negative=False, positive_semi_definite=False,
                 data_includes_cash=False, # affects _universe_subselect
                 ignore_shape_check=False # affects _universe_subselect
                 ):
        self.data = data
        self.use_last_available_time = use_last_available_time
        self.allow_nans = allow_nans
        self.compile_parameter = compile_parameter
        self.non_negative = non_negative
        self.positive_semi_definite = positive_semi_definite
        self.universe_maybe_noncash = None
        self.data_includes_cash = data_includes_cash
        self.ignore_shape_check = ignore_shape_check

    def _recursive_pre_evaluation(self, universe, backtest_times):
        # super()._recursive_pre_evaluation(universe, backtest_times)
        if self.compile_parameter:
            value = self.internal__recursive_values_in_time(
                t=backtest_times[0])
            self.parameter = cp.Parameter(value.shape if hasattr(value, "shape") else (),
                                          PSD=self.positive_semi_definite, nonneg=self.non_negative)          
        
        self.universe_maybe_noncash = universe if self.data_includes_cash else universe[:-1]

    def value_checker(self, result):
        """Ensure that only scalars or arrays without np.nan are returned.
        """

        if np.isscalar(result):
            if np.isnan(result) and not self.allow_nans:
                raise NaNError(
                    f"{self.__class__.__name__}._recursive_values_in_time result is a np.nan scalar."
                )
            else:
                return result

        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)) and not self.allow_nans:
                message = f"{self.__class__.__name__}._recursive_values_in_time result is an array with np.nan's."
                if hasattr(self.data, 'columns') and len(self.data.columns) == len(result):
                    message += "Specifically, the problem is with symbol(s): " + str(
                        self.data.columns[np.isnan(result)])
                raise NaNError(message)
            else:
                # we pass a copy because it can be accidentally overwritten
                return np.array(result)

        raise DataError(
            f"{self.__class__.__name__}._recursive_values_in_time result is not a scalar or array."
        )
        
    def _universe_subselect(self, data):
        """This function subselects from ``data`` the relevant universe.
        
        See github issue #106.
        
        If data is a pandas Series we subselect its index. If we fail
        we throw an error. If data is a pandas DataFrame (covariance, exposure matrix) 
        we try to subselect its index and columns. If we fail on either
        we ignore the failure, but if we fail on both we throw an error.
        If data is a numpy 1-d array we check that its length is the same as the 
        universe's.
        If it is a 2-d array we check that at least one dimension is the
        same as the universe's.
        If the universe is None we skip all checks. (We may revisit this choice.) This only happens
        if the DataEstimator instance is not part of a PolicyEstimator tree 
        (a usecase which we will probably drop).
        """
        
        if (self.universe_maybe_noncash is None) or self.ignore_shape_check:
            return data.values if hasattr(data, 'values') else data
        
        if isinstance(data, pd.Series):
            try:
                return data.loc[self.universe_maybe_noncash].values
            except KeyError:
                raise MissingAssetsError(
                f"The pandas Series found by {self.__class__.__name__} has index {data.index}"
                f" while the current universe {'minus cash' if not self.data_includes_cash else ''}"
                f" is {self.universe_maybe_noncash}. It was not possible to reconcile the two.")
        
        if isinstance(data, pd.DataFrame):
            try:
                return data.loc[self.universe_maybe_noncash, self.universe_maybe_noncash].values
            except KeyError:
                try:
                    return data.loc[:, self.universe_maybe_noncash].values
                except KeyError:
                    try:
                        return data.loc[self.universe_maybe_noncash, :].values
                    except KeyError:
                        pass
            raise MissingAssetsError(
                f"The pandas DataFrame found by {self.__class__.__name__} has index {data.index}"
                f" and columns {data.columns}"
                f" while the current universe {'minus cash' if not self.data_includes_cash else ''}"
                f" is {self.universe_maybe_noncash}. It was not possible to reconcile the two.")
        
        if isinstance(data, np.ndarray):
            dimensions = data.shape
            if not len(self.universe_maybe_noncash) in dimensions:
                raise MissingAssetsError(
                    f"The numpy array found by {self.__class__.__name__} has dimensions {data.shape}"
                    f" while the current universe {'minus cash' if not self.data_includes_cash else ''}" 
                    f" has size {len(self.universe_maybe_noncash)}.")
            return data
        
        # scalar
        return data

                            

    def internal__recursive_values_in_time(self, t, *args, **kwargs):
        """Internal method called by `self._recursive_values_in_time`."""

        # if self.data has values_in_time we use it
        if hasattr(self.data, "values_in_time"):
            tmp = self.data.values_in_time(t=t, *args, **kwargs)
            return self.value_checker(self._universe_subselect(tmp) )

        # if self.data is pandas and has datetime (first) index
        if (hasattr(self.data, "loc") and hasattr(self.data, "index")
            and (isinstance(self.data.index, pd.DatetimeIndex)
                 or (isinstance(self.data.index, pd.MultiIndex) and 
                     isinstance(self.data.index.levels[0], pd.DatetimeIndex)))):
            try:
                if self.use_last_available_time:
                    if isinstance(self.data.index, pd.MultiIndex):
                        newt = self.data.index.levels[0][
                            self.data.index.levels[0] <= t][-1]
                    else:
                        newt = self.data.index[self.data.index <= t][-1]
                    tmp = self.data.loc[newt]
                else:
                    tmp = self.data.loc[t]
                
                return self.value_checker(self._universe_subselect(tmp))


            except (KeyError, IndexError):
                raise MissingTimesError(
                    f"{self.__class__.__name__}._recursive_values_in_time could not find data for time {t}.")

        # if data is pandas but no datetime index (constant in time)
        if hasattr(self.data, "values"):
            return self.value_checker(self._universe_subselect(self.data))

        # if data is scalar or numpy
        return self.value_checker(self._universe_subselect(self.data))

    def _recursive_values_in_time(self, t, *args, **kwargs):
        """Obtain value of `self.data` at time t or right before.

        Args:
            t (pandas.TimeStamp): time at which we evaluate the estimator

        Returns:
            result (float, numpy.array): if you use a callable object make
                sure that it returns a float or numpy array (and not,
                for example, a pandas object)

        """
        self.current_value = self.internal__recursive_values_in_time(
            t, *args, **kwargs)
        if hasattr(self, 'parameter'):
            self.parameter.value = self.current_value
        return self.current_value

    def __repr__(self):
        if np.isscalar(self.data):
            return str(self.data)
        if hasattr(self.data, 'values_in_time'):
            return self.data.__repr__()
        return repr_numpy_pandas(self.data)

# class ConstantEstimator(cvxpy.Constant, DataEstimator):
#     """Cvxpy constant that uses the pre_evalution method to be initialized."""
#
#     def _recursive_pre_evaluation(self, returns, volumes, start_time, end_time, **kwargs):
#         """You should call super().__init__ it here."""
#         raise NotImplementedError


# class KnownData(DataEstimator):
#     """Data known beforehand to use in backtest.
#
#     :param data: user-provided data (known beforehand) in the form of time-indexed
#         Series or DataFrame (points-in-time are used in the backtest),
#         non-time indexed Series of DataFrame (treated as constant in time),
#         or float. If time-indexed, it can be multi-indexed (for example for
#         covariance matrices at points in time).
#     :type data: pd.DataFrame, pd.Series, float
#     :param use_last_available_time: if True, use last available time in the data
#         at each point in the backtest
#     :type use_last_available_time: bool
#
#     :ivar value: current point-in-time value populated during a backtest
#
#     :raises MissingValueError: if data retrieved at a point in time contains nan's
#     """
#
#     def __init__(self, data, use_last_available_time=False):
#         self.data = data
#         self.use_last_available_time = use_last_available_time
#
#     def _recursive_values_in_time(self, t, **kwargs):
#         self.value = self.internal__recursive_values_in_time(t, *args, **kwargs)
#
#
# class KnownDataParameter(KnownData):
#     """Data known beforehand to use in backtest as Cvxpy parameter.
#
#     :param data: user-provided data (known beforehand) in the form of time-indexed
#         Series or DataFrame (points-in-time are used in the backtest),
#         non-time indexed Series of DataFrame (treated as constant in time),
#         or float. If time-indexed, it can be multi-indexed (for example for
#         covariance matrices at points in time).
#     :type data: pd.DataFrame, pd.Series, float
#     :param use_last_available_time: if True, use last available time in the data
#         at each point in the backtest
#     :type use_last_available_time: bool
#     :param cvxpy: if True, creates a Cvxpy parameter as self.parameter
#         with the shape of the provided data at the first point in time, and optional
#         attributes positive_semi_definite and non_negative
#     :type cvxpy: bool
#     :param positive_semi_definite: if True, make a PSD Cvxpy parameter
#     :type positive_semi_definite: bool
#     :param non_negative: if True, make a non negative Cvxpy parameter
#     :type non_negative: bool
#
#     :ivar parameter: Cvxpy parameter (if :param cvxpy: is True) with
#         value equal to :ivar value:
#
#     :raises MissingValueError: if data retrieved at a point in time contains nan's
#     """
#
#     def __init__(self, data, positive_semi_definite=False, non_negative=False, **kwargs):
#         super().__init__(data, **kwargs)
#         self.positive_semi_definite = positive_semi_definite
#         self.non_negative = non_negative
#
#     def _recursive_pre_evaluation(self, universe, backtest_times):
#         value = super().internal__recursive_values_in_time(t=backtest_times[0])
#         self.parameter = cp.Parameter(
#             value if hasattr(value, "shape") else (),
#             PSD=self.positive_semi_definite, nonneg=self.non_negative)
#
#     def _recursive_values_in_time(self, t, **kwargs):
#         self.parameter.value = self.internal__recursive_values_in_time(t, **kwargs)


#
# class ParameterEstimator(DataEstimator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(self, *args, compile_parameter=True, **kwargs)

# class ParameterEstimator(cvxpy.Parameter, DataEstimator, PolicyEstimator):
#     """Data estimator of point-in-time values that contains a Cvxpy Parameter.
#
#     Attributes:
#         parameter (cvxpy.Parameter): the parameter object to use with cvxpy
#             expressions
#     Args:
#         same as cvxportfolio.DataEstimator
#
#     """
#
#     def __init__(self, data, positive_semi_definite=False, non_negative=False, use_last_available_time=False, allow_nans=False):
#         self.positive_semi_definite = positive_semi_definite
#         self.non_negative = non_negative
#         self.use_last_available_time = use_last_available_time
#         self.data = data
#         self.allow_nans = allow_nans
#         # super(DataEstimator).__init__(data, use_last_available_time)
#
#     def _recursive_pre_evaluation(self, universe, backtest_times):
#         """Use the start time of the simulation to initialize the Parameter."""
#         super()._recursive_pre_evaluation(universe, backtest_times)
#         value = super()._recursive_values_in_time(t=backtest_times[0])
#         super().__init__(value.shape if hasattr(value, "shape") else (),
#             PSD=self.positive_semi_definite, nonneg=self.non_negative)
#
#     def _recursive_values_in_time(self, t, **kwargs):
#         """Update Cvxpy Parameter value."""
#         self.value = super()._recursive_values_in_time(t=t, **kwargs)
