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

import unittest

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.errors import (DataError, MissingAssetsError,
                                 MissingTimesError, NaNError)
from cvxportfolio.estimator import DataEstimator


class PlaceholderCallable:
    """Simple class with callback."""

    # pylint: disable=too-few-public-methods
    # pylint: disable=missing-return-doc
    # pylint: disable=missing-return-type-doc
    # pylint: disable=missing-param-doc

    def __init__(self, value):
        self.value = value

    def values_in_time(self, *args, **kwargs):
        """Callback used when passing this to DataEstimator."""
        return self.value


class TestEstimator(unittest.TestCase):
    """Test base Estimator objects."""

    def test_callable(self):
        """Test DataEstimator with an internal Estimator."""
        estimator = DataEstimator(PlaceholderCallable(1.0))
        time = pd.Timestamp("2022-01-01")
        self.assertEqual(estimator.values_in_time_recursive(
            t=time, current_portfolio_value=1000), 1.0)

        estimator = DataEstimator(PlaceholderCallable(np.nan))
        with self.assertRaises(NaNError):
            estimator.values_in_time_recursive(
                t=time, current_portfolio_value=1000)

        data = np.arange(10.0)
        estimator = DataEstimator(PlaceholderCallable(data))
        self.assertTrue(
            np.all(estimator.values_in_time_recursive(
                t=time, current_portfolio_value=1000) == data))

        with self.assertRaises(ValueError):
            estimator.simulate_recursive(t=time, current_portfolio_value=1000)

        data[1] = np.nan
        with self.assertRaises(NaNError):
            estimator.values_in_time_recursive(
                t=time, current_portfolio_value=1000)

    def test_scalar(self):
        """Test DataEstimator with a scalar."""
        time = pd.Timestamp("2022-01-01")

        estimator = DataEstimator(1.0)
        self.assertTrue(estimator.values_in_time_recursive(t=time) == 1.0)
        estimator = DataEstimator(1)
        self.assertTrue(estimator.values_in_time_recursive(t=time) == 1.0)

        estimator = DataEstimator(np.nan)
        with self.assertRaises(NaNError):
            estimator.values_in_time_recursive(t=time)

    def test_array(self):
        """Test DataEstimator with a numpy array."""
        time = pd.Timestamp("2022-01-01")
        data = np.arange(10.0)

        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator.values_in_time_recursive(t=time) == data))

        data[1] = np.nan
        estimator = DataEstimator(data)
        with self.assertRaises(NaNError):
            estimator.values_in_time_recursive(t=time)

    def test_series_dataframe_notime(self):
        """Test DataEstimator with a dataframe not time-indexed."""
        time = pd.Timestamp("2022-01-01")
        data = pd.Series(np.arange(10.0))
        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator.values_in_time_recursive(t=time) == data.values))

        data = pd.DataFrame(np.random.randn(3, 3))
        estimator = DataEstimator(data)
        self.assertTrue(
            np.all(estimator.values_in_time_recursive(t=time) == data.values))

    def test_series_timeindex(self):
        """Test DataEstimator with time-indexed series."""
        index = pd.date_range("2022-01-01", "2022-01-30")
        print(index)
        data = pd.Series(np.arange(len(index)), index)
        estimator = DataEstimator(data)

        print(estimator.values_in_time_recursive(t="2022-01-05"))
        self.assertTrue(estimator.values_in_time_recursive(
            t="2022-01-05") == data.loc["2022-01-05"])

        with self.assertRaises(MissingTimesError):
            estimator.values_in_time_recursive(t="2022-02-05")

        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(estimator.values_in_time_recursive(
            t="2022-02-05") == data.iloc[-1])

        with self.assertRaises(MissingTimesError):
            estimator.values_in_time_recursive(t="2021-02-05")

        data["2022-01-05"] = np.nan
        estimator = DataEstimator(data)
        self.assertTrue(estimator.values_in_time_recursive(
            t="2022-01-04") == data.loc["2022-01-04"])
        with self.assertRaises(NaNError):
            estimator.values_in_time_recursive(t="2022-01-05")

        estimator = DataEstimator('hello')
        with self.assertRaises(DataError):
            estimator.values_in_time_recursive(t="2022-01-05")

    def test_dataframe_timeindex(self):
        """Test DataEstimator with time-indexed dataframe."""
        index = pd.date_range("2022-01-01", "2022-01-30")
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data)

        print(estimator.values_in_time_recursive(t="2022-01-05"))
        self.assertTrue(np.all(estimator.values_in_time_recursive(
            t="2022-01-05") == data.loc["2022-01-05"]))

        with self.assertRaises(MissingTimesError):
            estimator.values_in_time_recursive(t="2021-01-05")

        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(
            np.all(estimator.values_in_time_recursive(t="2022-02-05"
                ) == data.iloc[-1]))

        data.loc["2022-01-05", 3] = np.nan
        estimator = DataEstimator(data, use_last_available_time=True)
        with self.assertRaises(MissingTimesError):
            estimator.values_in_time_recursive(t="2021-01-05")

    def test_series_notime_assetselect(self):
        """Test _universe_subselect."""
        universe = ['a', 'b', 'c']
        t = pd.Timestamp('2000-01-01')

        # data includes cash acct
        data = pd.Series(range(len(universe)), index=universe)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=universe, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data.values)

        # data excludes cash acct
        data = pd.Series(range(len(universe)), index=universe)
        estimator = DataEstimator(data)
        estimator.initialize_estimator_recursive(
            universe=universe, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data.values[:2])

        # shuffled universe
        estimator = DataEstimator(data.iloc[::-1])
        estimator.initialize_estimator_recursive(
            universe=universe, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data.values[:2])

        # wrong universe
        data = pd.Series(range(len(universe)), index=universe)
        estimator = DataEstimator(data)
        estimator.initialize_estimator_recursive(
            universe=['d', 'e', 'f'], trading_calendar=[t])
        with self.assertRaises(MissingAssetsError):
            result = estimator.values_in_time_recursive(t=t)

        # selection of universe
        data = pd.Series(range(len(universe)), index=universe)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=['b'], trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data.values[1])

    def test_ndarray_assetselect(self):
        """Test errors if ndarray is not of right size."""
        data = np.zeros((2, 3))
        t = pd.Timestamp('2000-01-01')

        # with universe of size 2
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=['a', 'b'], trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data)

        # with universe of size 3
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=['a', 'b', 'c'], trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data)

        # error with universe of size 4
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=['a', 'b', 'c', 'd'], trading_calendar=[t])
        with self.assertRaises(MissingAssetsError):
            result = estimator.values_in_time_recursive(t=t)

        # all ok if skipping check
        estimator = DataEstimator(data, data_includes_cash=True,
            ignore_shape_check=True)
        estimator.initialize_estimator_recursive(
            universe=['a', 'b', 'c', 'd'], trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        assert np.all(result == data)

    def test_dataframe_multindex(self):
        """Test DataEstimator with a multi-indexed dataframe.

        We also check that _universe_subselect works fine.
        """
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        print(data.index)
        estimator = DataEstimator(data)
        self.assertTrue(np.all(estimator.values_in_time_recursive(
            t="2022-01-05") == data.loc["2022-01-05"]))

        # use_last_avalaible_time
        estimator = DataEstimator(data, use_last_available_time=True)
        self.assertTrue(np.all(estimator.values_in_time_recursive(
            t="2022-02-05") == data.loc["2022-01-30"]))
        self.assertTrue(np.all(estimator.values_in_time_recursive(
            t="2022-01-05") == data.loc["2022-01-05"]))
        with self.assertRaises(MissingTimesError):
            estimator.values_in_time_recursive(t="2020-01-05")

        # universe subselect
        t = "2022-01-01"
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=second_level, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        self.assertTrue(np.all(result == data.loc[t]))

        # result has same second_level as columns
        data = pd.DataFrame(np.random.randn(len(index), len(second_level)),
                            index=index, columns=second_level)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=second_level, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        self.assertTrue(np.all(result == data.loc[t]))

        # universe are columns
        uni = ['a', 'b']
        data = pd.DataFrame(np.random.randn(len(index), 2),
                            index=index, columns=uni)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=uni, trading_calendar=[t])
        result = estimator.values_in_time_recursive(t=t)
        self.assertTrue(np.all(result == data.loc[t]))

        # wrong universe
        data = pd.DataFrame(np.random.randn(len(index), 2),
                            index=index, columns=uni)
        estimator = DataEstimator(data, data_includes_cash=True)
        estimator.initialize_estimator_recursive(
            universe=uni + ['c'], trading_calendar=[t])
        with self.assertRaises(MissingAssetsError):
            result = estimator.values_in_time_recursive(t=t)

        # if timeindex is not first level it is not picked up
        index = pd.MultiIndex.from_product([second_level, timeindex])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data)
        assert np.all(estimator.values_in_time_recursive(
            t="2020-01-05") == data.values)

    def test_parameter_estimator(self):
        """Test DataEstimator's cvxpy parameter."""
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        estimator = DataEstimator(data, compile_parameter=True,
            data_includes_cash=True)
        self.assertTrue(estimator.parameter is None)
        estimator.initialize_estimator_recursive(
            universe=data.columns, trading_calendar=timeindex)
        # assert hasattr(estimator, 'parameter')
        self.assertTrue(estimator.parameter is not None)
        estimator.values_in_time_recursive(t="2022-01-05")
        self.assertTrue(
            np.all(estimator.parameter.value == data.loc["2022-01-05"]))

    def test_repr_dataestimator(self):
        """Test __repr__ magic method of DataEstimator."""
        print(DataEstimator(3))
        print(DataEstimator(np.array([1, 2, 3])))
        print(DataEstimator(pd.Series([1, 2, 3])))
        print(DataEstimator(pd.DataFrame([1, 2, 3])))

    def test_repr(self):
        """Test other assorted __repr__ methods of derived objects."""
        print(cvx.FactorModelCovariance(num_factors=10))
        print(cvx.ReturnsForecast() - .5 * cvx.FullCovariance())
        print(cvx.SinglePeriodOptimization(cvx.ReturnsForecast(),
                                           [cvx.LongOnly()]))
        print(cvx.LeverageLimit(3))


if __name__ == '__main__':

    unittest.main() # pragma: no cover
