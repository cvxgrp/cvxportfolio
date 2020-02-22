"""
Copyright 2020 Enzo Busseti.

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

import pandas as pd


logger = logging.getLogger(__name__)

__all__ = ['values_in_time']


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

        # if isinstance(obj.index, pd.MultiIndex):
        #     previous_tau = obj.loc[:tau, :].index.values[-1][0]
        #     if return_all_ts:
        #         return obj.loc[previous_tau]
        #     else:
        #         return obj.loc[previous_tau].loc[t]
        # else:
        #     previous_tau = obj.loc[:tau, :].index.values[-1]
        #     return obj.loc[prev_tau, :]

    return obj
