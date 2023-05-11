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
This module contains classes that have been replaced by improved versions.
They are kept because they have some slight differences and are used by 
the original examples scripts to generate the book's plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import logging
import cvxpy as cvx
import numpy as np

from .returns import BaseReturnsModel, MultipleReturnsForecasts
from .estimator import ParameterEstimator

logger = logging.getLogger(__name__)

__all__ = [
    "null_checker",
    "non_null_data_args",
    "values_in_time",
    "plot_what_if",
    "LegacyReturnsForecast",
    "MPOReturnsForecast",
    ]


def values_in_time(obj, t, tau=None):
    """Obtain value(s) of object at time t, or right before.

    Optionally specify time tau>=t for which we want a prediction,
    otherwise it is assumed tau = t.

    obj: callable, pd.Series, pd.DataFrame, or something else.

        If a callable, we return obj(t,tau).

        If obj has an index attribute,
        we try to return obj.loc[t],
        or obj.loc[t, tau], if the index is a MultiIndex.
        If not available, we return obj.

        Otherwise, we return obj.

    t: np.Timestamp (or similar). Time at which we want
        the value.

    tau: np.Timestamp (or similar), or None. Time tau >= t
        of the prediction,  e.g., tau could be tomorrow, t
        today, and we ask for prediction of market volume tomorrow,
        made today. If None, then it is assumed tau = t.

    """

    if hasattr(obj, "__call__"):
        return obj(t, tau)

    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        try:
            if not (tau is None) and isinstance(obj.index, pd.MultiIndex):
                return obj.loc[(t, tau)]
            else:
                return obj.loc[t]
        except KeyError:
            return obj

    return obj


def plot_what_if(time, true_results, alt_results):
    true_results.value.plot(label=true_results.pol_name)
    for result in alt_results:
        result.value.plot(label=result.pol_name, linestyle="--")
    plt.axvline(x=time, linestyle=":")


def null_checker(obj):
    """Check if obj contains NaN."""
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        if np.any(pd.isnull(obj)):
            raise ValueError("Data object contains NaN values", obj)
    elif np.isscalar(obj):
        if np.isnan(obj):
            raise ValueError("Data object contains NaN values", obj)
    else:
        raise TypeError("Data object can only be scalar or Pandas.")


def non_null_data_args(f):
    def new_f(*args, **kwds):
        for el in args:
            null_checker(el)
        for el in kwds.values():
            null_checker(el)
        return f(*args, **kwds)

    return new_f


# LEGACY CLASSES USED BY OLD TESTS. WILL BE REMOVED AS WE FINISH TRANSLATION


class LegacyReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    STILL USED BY OLD PARTS OF CVXPORTFOLIO

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0.0, gamma_decay=None, name=None):
        #null_checker(returns)
        self.returns = ParameterEstimator(returns)
        #null_checker(delta)
        self.delta = ParameterEstimator(delta)
        self.gamma_decay = gamma_decay
        self.name = name
        
    def compile_to_cvxpy(self, wplus, z, value):
        alpha = cvx.multiply(self.returns, wplus)
        alpha -= cvx.multiply(self.delta, cvx.abs(wplus))
        return cvx.sum(alpha)

    # def weight_expr(self, t, wplus, z=None, v=None):
    #     """Returns the estimated alpha.
    #
    #     Args:
    #       t: time estimate is made.
    #       wplus: An expression for holdings.
    #       tau: time of alpha being estimated.
    #
    #     Returns:
    #       An expression for the alpha.
    #     """
    #     alpha = cvx.multiply(values_in_time(self.returns, t), wplus)
    #     alpha -= cvx.multiply(values_in_time(self.delta, t), cvx.abs(wplus))
    #     return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data):
        self.alpha_data = alpha_data

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        return self.alpha_data[(t, tau)].values.T * wplus