# Copyright 2023- The Cvxportfolio Contributors
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
from .data import MissingValuesError, DataError


class Estimator:
    """Estimator base class.

    Policies, costs, and constraints inherit from this.
    """

    children = []

    def prescient_evaluation(self, returns, volumes, **kwargs):
        """Cache computation with prescience recursively on its children.

        This function is called by Simulator classes before the
        start of a backtest with the full dataset available to the
        simulator. This is useful for estimators such as pandas.rolling_mean
        which are faster (and easier) when vectorized rather than called separately
        at each point in time.
        
        It should be used very carefully, if you
        are not sure do not implement this method. You can use 
        values_in_time to build lazily whatever you would 
        build here beforehand. If you do implement this, double 
        check that at each point in time only past data
        (with respect to that point) is used and not future data.

        Args:
            returns (pandas.DataFrame): market returns
            volumes (pandas.DataFrame): market volumes
            kwargs (dict): extra data available
        """
        for child in children:
            child.prescient_evaluation(returns, volumes, **kwargs)

    def values_in_time(self, t, **kwargs):
        """Evaluates estimator at a point in time recursively on its children.

        This function is called by Simulator classes on Policy classes
        returning the current trades list. Policy classes, if
        they contain internal estimators, should register them
        in `self.children`  and call this base function before
        they do their internal computation. CvxpyExpression estimators
        should instead define this method to update their Cvxpy parameters.

        Args:
            t (pd.TimeStamp): point in time of the simulation
            kwargs (dict): extra data used
        """
        for child in children:
            child.values_in_time(t, current_portfolio, **kwargs) 
        


class CvxpyExpressionEstimator(Estimator):
    """Base class for estimators that are Cvxpy expressions."""

    def compile_to_cvxpy(self, w_plus, z):
        """Compile term to cvxpy expression.

        This is called by a Policy class on its terms before the start of the backtest
        to compile its Cvxpy problem. If the Policy changes in time
        this is called at every time step.

        It can either return a scalar expression, in the case of objective terms,
        or a list of cvxpy constraints, in the case of constraints.

        In MultiPeriodOptimization policies this is called separately
        for costs and constraints at different look-ahead steps with
        the corresponding w_plus and z.

        Args:
            w_plus (cvxpy.Variable): post-trade allocation weights vector
            z (cvxpy.Variable): trades weight vector

        Returns:
            cvxpy.Expression
        """
        raise NotImplementedError


class DataEstimator(Estimator):
    """Estimator of point-in-time values from internal `self.data`.
    
        Args:
            data (object, pandas.Series, pandas.DataFrame): Data expressed
                preferably as pandas Series or DataFrame where the first
                index is a pandas.DateTimeIndex. Otherwise you can
                set it as a callable object which takes a pandas.TimeStamp
                and returns the corresponding value in time, or a constant
                float, numpy.array, or even pandas Series or DataFrame not indexed
                by time (e.g., a covariance matrix where both index and columns
                are the stock symbols).
            use_last_available_time (bool): if the pandas index exists 
                and is a pandas.DateTimeIndex you can instruct self.values_in_time
                to retrieve the last available value at time t by setting 
                this to True. Default is False.

    """
    
    def __init__(self, data, use_last_available_time=False):
        self.data = data
        self.use_last_available_time = use_last_available_time
        

    def value_checker(self, result):
        """Ensure that only scalars or arrays without np.nan are returned.
        
        Args: 
            result (int, float, or np.array): data produced by self.values_in_time
        
        Returns:
            result (int, float, or np.array): same data if no np.nan are present and type is correct
        
        Raises:
            cvxportfolio.MissingValuesError: if np.nan's are present in result
            cvxportfolio.DataError: if data is not in the right form
        """
        
        if np.isscalar(result):
            if np.isnan(result):
                raise MissingValuesError(f"{self.__class__.__name__}.values_in_time result is a np.nan scalar.")
            else:
                return result
            
        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)):
                raise MissingValuesError(f"{self.__class__.__name__}.values_in_time result is an array with np.nan's.")
            else:
                return result
                
        raise DataError(f"{self.__class__.__name__}.values_in_time result is not a scalar or array.")
        
              
    def values_in_time(self, t, **kwargs):
        """Obtain value of `self.data` at time t or right before.
        
        Args:
            t (pandas.TimeStamp): time at which we evaluate the estimator 
        
        Returns:
            result (float, numpy.array): if you use a callable object make
                sure that it returns a float or numpy array (and not,
                for example, a pandas object)

        """

        if hasattr(self.data, "__call__"):
            return self.value_checker(self.data(t, **kwargs))
            
        if hasattr(self.data, 'loc') and hasattr(self.data, 'index') and (
        isinstance(self.data.index, pd.DatetimeIndex) or
        (isinstance(self.data.index, pd.MultiIndex) and 
        isinstance(self.data.index.levels[0], pd.DatetimeIndex))
        ):
            try:
                if self.use_last_available_time:
                    tmp = self.data.loc[self.data.index <= t].iloc[-1]
                else:
                    tmp = self.data.loc[t]
                if hasattr(tmp, 'values'):
                    return self.value_checker(tmp.values)
                else:
                    return self.value_checker(tmp)
                
            except (KeyError, IndexError):
                raise MissingValuesError(f"{self.__class__.__name__}.values_in_time could not find data for requested time.")

        if hasattr(self.data, 'values'):
            return self.value_checker(self.data.values)
            
        return self.value_checker(self.data)
        
