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

"""Unit tests for the data and parameter estimator objects."""

import numpy as np
import pandas as pd
import unittest

from cvxportfolio.estimator import DataEstimator  # , ParameterEstimator
from cvxportfolio.errors import MissingValuesError, DataError
import cvxportfolio as cvx


class PlaceholderCallable:
    def __init__(self, value):
        self.value = value

    def values_in_time(self, t, **kwargs):
        return self.value


class TestEstimators(unittest.TestCase):

    def test_callable(self):
        estimator = DataEstimator(PlaceholderCallable(1.0))
        time = pd.Timestamp("2022-01-01")
        self.assertEqual(estimator._recursive_values_in_time(time), 1.0)

        estimator = DataEstimator(PlaceholderCallable(np.nan))
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time(t=time)

        data = np.arange(10.0)
        estimator = DataEstimator(PlaceholderCallable(data))
        self.assertTrue(
            np.all(estimator._recursive_values_in_time(t=time) == data))

        data[1] = np.nan
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time(time)

    def test_scalar(self):
        time = pd.Timestamp("2022-01-01")

        estimator = DataEstimator(1.0)
        self.assertTrue(estimator._recursive_values_in_time(t=time) == 1.0)
        estimator = DataEstimator(1)
        self.assertTrue(estimator._recursive_values_in_time(t=time) == 1.0)

        estimator = DataEstimator(np.nan)
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time(t=time)

    def test_array(self):
        time = pd.Timestamp("2022-01-01")
        data = np.arange(10.0)

        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator._recursive_values_in_time(t=time) == data))

        data[1] = np.nan
        estimator = DataEstimator(data)
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time(t=time)

    def test_series_dataframe_notime(self):
        time = pd.Timestamp("2022-01-01")
        data = pd.Series(np.arange(10.0))
        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator._recursive_values_in_time(t=time) == data.values))

        data = pd.DataFrame(np.random.randn(3, 3))
        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator._recursive_values_in_time(t=time) == data.values))

    def test_series_timeindex(self):
        index = pd.date_range("2022-01-01", "2022-01-30")
        print(index)
        data = pd.Series(np.arange(len(index)), index)
        estimator = DataEstimator(data)

        print(estimator._recursive_values_in_time("2022-01-05"))
        self.assertTrue(estimator._recursive_values_in_time(
            "2022-01-05") == data.loc["2022-01-05"])

        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2022-02-05")

        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(estimator._recursive_values_in_time(
            "2022-02-05") == data.iloc[-1])

        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2021-02-05")

        data["2022-01-05"] = np.nan
        estimator = DataEstimator(data)
        self.assertTrue(estimator._recursive_values_in_time(
            "2022-01-04") == data.loc["2022-01-04"])
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2022-01-05")

    def test_dataframe_timeindex(self):
        index = pd.date_range("2022-01-01", "2022-01-30")
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data)

        print(estimator._recursive_values_in_time("2022-01-05"))
        self.assertTrue(np.all(estimator._recursive_values_in_time(
            "2022-01-05") == data.loc["2022-01-05"]))

        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2021-01-05")

        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(
            np.all(estimator._recursive_values_in_time("2022-02-05") == data.iloc[-1]))

        data.loc["2022-01-05", 3] = np.nan
        estimator = DataEstimator(data, use_last_available_time=True)
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2021-01-05")

    def test_dataframe_multindex(self):
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        print(data.index)
        estimator = DataEstimator(data)
        self.assertTrue(np.all(estimator._recursive_values_in_time(
            "2022-01-05") == data.loc["2022-01-05"]))

        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(np.all(estimator._recursive_values_in_time(
            "2022-02-05") == data.loc["2022-01-30"]))
        self.assertTrue(np.all(estimator._recursive_values_in_time(
            "2022-01-05") == data.loc["2022-01-05"]))
        with self.assertRaises(MissingValuesError):
            estimator._recursive_values_in_time("2020-01-05")

        index = pd.MultiIndex.from_product([second_level, timeindex])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data)
        assert np.all(estimator._recursive_values_in_time(
            "2020-01-05") == data.values)

    def test_parameter_estimator(self):
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data, compile_parameter=True)
        self.assertTrue(not hasattr(estimator, "parameter"))
        estimator._recursive_pre_evaluation(
            universe=None, backtest_times=timeindex)
        # assert hasattr(estimator, 'parameter')
        self.assertTrue(hasattr(estimator, "parameter"))
        estimator._recursive_values_in_time("2022-01-05")
        self.assertTrue(
            np.all(estimator.parameter.value == data.loc["2022-01-05"]))

    def test_repr_dataestimator(self):
        print(DataEstimator(3))
        print(DataEstimator(np.array([1, 2, 3])))
        print(DataEstimator(pd.Series([1, 2, 3])))
        print(DataEstimator(pd.DataFrame([1, 2, 3])))

    def test_repr(self):
        print(cvx.FactorModelCovariance(num_factors=10))
        print(cvx.ReturnsForecast() - .5 * cvx.FullCovariance())
        print(cvx.SinglePeriodOptimization(cvx.ReturnsForecast(),
                                           [cvx.LongOnly(nocash=True)]))
        print(cvx.LeverageLimit(3))


if __name__ == '__main__':
    unittest.main()
