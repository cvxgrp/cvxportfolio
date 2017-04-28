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

import numpy as np
import pandas as pd

__all__ = ['matrix_locator', 'vector_locator', 'null_checker']


def null_checker(obj):
    """Check if obj contains NaN."""
    if isinstance(obj, pd.Panel) or isinstance(obj, pd.DataFrame) \
        or isinstance(obj, pd.Series):
        return np.any(pd.isnull(obj))
    elif np.isscalar(obj):
        return np.isnan(obj)
    else:
        raise TypeError('Data can only be scalar or a Pandas object.')


def matrix_locator(obj, t):
    """Retrieve a matrix from a time indexed Panel, or a static DataFrame."""
    if isinstance(obj, pd.Panel):
        return obj.iloc[obj.axes[0].get_loc(t, method='pad')]
    elif isinstance(obj, pd.DataFrame):
        return obj
    else: # obj not pandas
        raise TypeError('Expected Pandas DataFrame or Panel, got:', obj)

def vector_locator(obj, t, n):
    """Retrieve vector of size n from a time indexed DF, a Series or scalar."""
    if isinstance(obj, pd.DataFrame):
        res=obj.iloc[obj.axes[0].get_loc(t, method='pad')]
        assert(len(res)==n)
        return res
    elif isinstance(obj, pd.Series):
        assert(len(obj)==n)
        return obj
    elif np.isscalar(obj):
        return np.ones(n)*obj
    else: # obj not pandas
        raise TypeError('Expected Pandas DataFrame, Series, or scalar. Got:', obj)
