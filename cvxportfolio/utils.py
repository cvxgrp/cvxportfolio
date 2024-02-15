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
"""This module contains miscellaneous functions."""

import hashlib
from numbers import Number

import numpy as np
import pandas as pd

from .errors import DataError

TRUNCATE_REPR_HASH = 10  # probability of conflict is 1e-16


__all__ = ['periods_per_year_from_datetime_index', 'resample_returns',
           'flatten_heterogeneous_list', 'repr_numpy_pandas',
           'average_periods_per_year']


def set_pd_read_only(df_or_ser):
    """Set numpy array contained in dataframe or series to read only.

    This is done on data store internally before it is served to the
    policy or the simulator to ensure data consistency in case some
    element of the pipeline accidentally corrupts the data.

    This is enough to prevent direct assignement to the resulting
    dataframe. However it could still be accidentally corrupted by
    assigning to columns or indices that are not present in the
    original. We avoid that case as well by returning a wrapped
    dataframe (which doesn't copy data on creation) in
    serve_data_policy and serve_data_simulator.

    :param df_or_ser: Series or Dataframe, only numeric (better if
        homogeneous) dtype.
    :type df_or_ser: pd.Series or pd.DataFrame

    :returns: Pandas object set to read only.
    :rtype: pd.Series or pd.DataFrame
    """
    data = df_or_ser.values
    data.flags.writeable = False
    if hasattr(df_or_ser, 'columns'):
        return pd.DataFrame(data, index=df_or_ser.index,
                            columns=df_or_ser.columns)
    return pd.Series(data, index=df_or_ser.index, name=df_or_ser.name)


def average_periods_per_year(num_periods, first_time, last_time):
    """Average periods per year of a datetime index (unpacked), rounded to int.

    :param num_periods: Length of the index.
    :type num_periods: int
    :param first_time: First timestamp in the index.
    :type first_time: pandas.Timestamp
    :param last_time: Last timestamp in the index.
    :type last_time: pandas.Timestamp

    :returns: Average number of periods in a year, as implied by the index.
    :rtype: int
    """
    return int(np.round(num_periods / ((last_time - first_time) /
                                    pd.Timedelta('365.24d'))))

def periods_per_year_from_datetime_index(idx):
    """Average periods per year of a datetime index, rounded to int.

    :param idx: Datetime index, *e.g.*, of a returns dataframe.
    :type idx: pandas.DatetimeIndex

    :returns: Average number of periods in a year, as implied by the index.
    :rtype: int
    """
    return average_periods_per_year(
        num_periods=len(idx), first_time=idx[0], last_time=idx[-1])


def resample_returns(returns, periods):
    """Resample returns expressed over number of periods to single period.

    :param returns: Original returns whith are, *e.g.*, annualized, and we
        want to get as daily.
    :type returns: pandas.Series
    :param periods: Number of trading periods (*e.g.*, trading days) in the
         period of the returns (*e.g.*, 252 if annualized).
    :type periods: int

    :returns: Resampled returns.
    :rtype: pandas.Series
    """
    return np.exp(np.log(1 + returns) / periods) - 1

def make_numeric(np_or_pd):
    """Coerce Pandas or Numpy object to numeric.

    :param np_or_pd: User-provided data.
    :type np_or_pd: np.array, pd.Series, pd.DataFrame, object

    :raises DataError: If input data could not be casted to numeric.

    :returns: Same object, casted to numeric if necessary
    :rtype: np.array, pd.Series, pd.DataFrame, object
    """

    try:
        if isinstance(np_or_pd, np.ndarray):
            if not np.issubdtype(np_or_pd.dtype, np.number):
                return np_or_pd.astype(float)

        if isinstance(np_or_pd, pd.Series):
            if not np.issubdtype(np_or_pd.dtype, np.number):
                return pd.to_numeric(np_or_pd)

        if isinstance(np_or_pd, pd.DataFrame):
            if not np.all(
                [np.issubdtype(el, np.number) for el in set(np_or_pd.dtypes)]):
                return np_or_pd.astype(float)

    except ValueError as exc:
        raise DataError("Input data could not be cast to numeric.") from exc

    return np_or_pd


def flatten_heterogeneous_list(li):
    """[1, 2, 3, [4, 5]] -> [1, 2, 3, 4, 5].

    :param li: List that may contain lists, which we concatenate.
    :type li: list

    :returns: Flattened list.
    :rtype: list
    """
    return sum(([el] if not hasattr(el, '__iter__')
                else el for el in li), [])


def hash_(array_like):
    """Hash np.array.

    :param array_like: Array object of which we obtain a hash.
    :type array_like: numpy.array

    :returns: Unique hash.
    :rtype: str
    """
    return hashlib.sha256(
        bytes(str(list(array_like.flatten())), 'utf-8')).hexdigest()[
            :TRUNCATE_REPR_HASH]


def repr_numpy_pandas(nppd):
    """Unique repr of a numpy or pandas object.

    :param nppd: Numpy or pandas object of which we obtain a unique hash.
    :type nppd: numpy.array, pandas.Series, pandas.DataFrame

    :raises NotImplementedError: If the object is not supported.

    :returns: Unique hash.
    :rtype: str
    """
    if isinstance(nppd, np.ndarray):
        return f'np.array(hash={hash_(nppd)})'
    # if isinstance(nppd, np.matrix):
    #     return f'np.array(hash={hash_(nppd.A)})'
    if isinstance(nppd, pd.Series):
        return f'pd.Series(hash_values={hash_(nppd.values)}, '\
            + f'hash_index={hash_(nppd.index.to_numpy())})'
    if isinstance(nppd, pd.DataFrame):
        return f'pd.DataFrame(hash_values={hash_(nppd.values)}, '\
            + f'hash_index={hash_(nppd.index.to_numpy())}, '\
            + f'hash_columns={hash_(nppd.columns.to_numpy())})'
    raise NotImplementedError('The provided data type is not supported.')
