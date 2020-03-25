"""
Copyright 2017 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

__all__ = ['null_checker', 'non_null_data_args',
           'values_in_time', 'plot_what_if']


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

    if hasattr(obj, '__call__'):
        return obj(t, tau)

    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        try:
            if isinstance(obj.index, pd.MultiIndex):
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
    if (isinstance(obj, pd.DataFrame) or
            isinstance(obj, pd.Series)):
        if np.any(pd.isnull(obj)):
            raise ValueError('Data object contains NaN values', obj)
    elif np.isscalar(obj):
        if np.isnan(obj):
            raise ValueError('Data object contains NaN values', obj)
    else:
        raise TypeError('Data object can only be scalar or Pandas.')


def non_null_data_args(f):
    def new_f(*args, **kwds):
        for el in args:
            null_checker(el)
        for el in kwds.values():
            null_checker(el)
        return f(*args, **kwds)
    return new_f
