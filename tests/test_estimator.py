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

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio.estimator import DataEstimator, ParameterEstimator
from cvxportfolio.errors import MissingValuesError, DataError


class PlaceholderCallable:
    def __init__(self, value):
        self.value = value

    def values_in_time(self, t, *args, **kwargs):
        return self.value


def test_callable():
    estimator = DataEstimator(PlaceholderCallable(1.0))
    time = pd.Timestamp("2022-01-01")
    assert estimator.values_in_time(time) == 1.0

    estimator = DataEstimator(PlaceholderCallable(np.nan))
    with pytest.raises(MissingValuesError):
        estimator.values_in_time(time)

    data = np.arange(10.0)
    estimator = DataEstimator(PlaceholderCallable(data))
    assert np.all(estimator.values_in_time(time) == data)

    data[1] = np.nan
    with pytest.raises(MissingValuesError):
        estimator.values_in_time(time)


def test_scalar():
    time = pd.Timestamp("2022-01-01")

    estimator = DataEstimator(1.0)
    assert estimator.values_in_time(time) == 1.0
    estimator = DataEstimator(1)
    assert estimator.values_in_time(time) == 1.0

    estimator = DataEstimator(np.nan)
    with pytest.raises(MissingValuesError):
        estimator.values_in_time(time)


def test_array():
    time = pd.Timestamp("2022-01-01")
    data = np.arange(10.0)

    estimator = DataEstimator(data)
    assert np.all(estimator.values_in_time(time) == data)

    data[1] = np.nan
    estimator = DataEstimator(data)
    with pytest.raises(MissingValuesError):
        estimator.values_in_time(time)


def test_series_dataframe_notime():
    time = pd.Timestamp("2022-01-01")
    data = pd.Series(np.arange(10.0))
    estimator = DataEstimator(data)
    assert np.all(estimator.values_in_time(time) == data.values)

    data = pd.DataFrame(np.random.randn(3, 3))
    estimator = DataEstimator(data)
    assert np.all(estimator.values_in_time(time) == data.values)


def test_series_timeindex():
    index = pd.date_range("2022-01-01", "2022-01-30")
    print(index)
    data = pd.Series(np.arange(len(index)), index)
    estimator = DataEstimator(data)

    print(estimator.values_in_time("2022-01-05"))
    assert estimator.values_in_time("2022-01-05") == data.loc["2022-01-05"]

    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2022-02-05")

    estimator = DataEstimator(data, use_last_available_time=True)
    assert estimator.values_in_time("2022-02-05") == data.iloc[-1]

    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2021-02-05")

    data["2022-01-05"] = np.nan
    estimator = DataEstimator(data)
    assert estimator.values_in_time("2022-01-04") == data.loc["2022-01-04"]
    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2022-01-05")


def test_dataframe_timeindex():
    index = pd.date_range("2022-01-01", "2022-01-30")
    data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
    estimator = DataEstimator(data)

    print(estimator.values_in_time("2022-01-05"))
    assert np.all(estimator.values_in_time("2022-01-05") == data.loc["2022-01-05"])

    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2021-01-05")

    estimator = DataEstimator(data, use_last_available_time=True)
    assert np.all(estimator.values_in_time("2022-02-05") == data.iloc[-1])

    data.loc["2022-01-05", 3] = np.nan
    estimator = DataEstimator(data, use_last_available_time=True)
    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2021-01-05")


def test_dataframe_multindex():
    timeindex = pd.date_range("2022-01-01", "2022-01-30")
    second_level = ["hello", "ciao", "hola"]
    index = pd.MultiIndex.from_product([timeindex, second_level])
    data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
    print(data.index)
    estimator = DataEstimator(data)
    assert np.all(estimator.values_in_time("2022-01-05") == data.loc["2022-01-05"])

    estimator = DataEstimator(data, use_last_available_time=True)
    assert np.all(estimator.values_in_time("2022-02-05") == data.loc["2022-01-30"])
    assert np.all(estimator.values_in_time("2022-01-05") == data.loc["2022-01-05"])
    with pytest.raises(MissingValuesError):
        estimator.values_in_time("2020-01-05")

    index = pd.MultiIndex.from_product([second_level, timeindex])
    data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
    estimator = DataEstimator(data)
    assert np.all(estimator.values_in_time("2020-01-05") == data.values)


def test_parameter_estimator():
    timeindex = pd.date_range("2022-01-01", "2022-01-30")
    second_level = ["hello", "ciao", "hola"]
    index = pd.MultiIndex.from_product([timeindex, second_level])
    data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
    estimator = ParameterEstimator(data)
    assert not hasattr(estimator, "value")
    estimator.pre_evaluation(
        returns=None, volumes=None, start_time="2022-01-01", end_time=None
    )
    # assert hasattr(estimator, 'parameter')
    assert hasattr(estimator, "value")
    estimator.values_in_time("2022-01-05")
    assert np.all(estimator.value == data.loc["2022-01-05"])
