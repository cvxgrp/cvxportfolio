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
"""Unit tests for some utils.py functions."""

import unittest

import numpy as np
import pandas as pd

from cvxportfolio.utils import make_numeric, repr_numpy_pandas


class TestUtils(unittest.TestCase):
    """Test functions in the cvxportfolio.utils module."""

    def test_hasher(self):
        """Test hashing function."""
        a = np.array([1, 2, 3])
        re = repr_numpy_pandas(a)
        print(re)
        b = np.array([[1, 2, 3]])
        re1 = repr_numpy_pandas(b)
        print(re1)
        self.assertTrue(re == re1)

        with self.assertRaises(NotImplementedError):
            repr_numpy_pandas('Hello!')

        a = pd.Series([1, 2, 3], ['2020-01-01', '2021-01-01', '2022-01-01'])
        re = repr_numpy_pandas(a)
        print(re)
        b = pd.Series(a.values, a.index)
        re1 = repr_numpy_pandas(b)
        print(re1)
        self.assertTrue(re == re1)

        # multiindexed df
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        data.columns = ["one", "two", "tre", "quattro",
                        "cinque", "sei", "sette", "otto", "nove", "dieci"]

        re = repr_numpy_pandas(data)
        print(re)

        data1 = pd.DataFrame(data.values,
                             index=data.index, columns=data.columns)

        re1 = repr_numpy_pandas(data1)
        print(re1)
        self.assertTrue(re == re1)

    def test_make_numeric(self):
        """Test make_numeric function for user-provided data."""

        for data in [
                2., np.array([1, 2]), pd.Series([1, 2, 3]),
                pd.DataFrame([[1, 2., 3], [4, 5., 6]])]:
            self.assertTrue(make_numeric(data) is data)

        for data in [
                np.array([1, 2], dtype=object), pd.Series([1, 2, 3], dtype=object),
                pd.DataFrame([[1, 2., 3], [4, 5., 6]], dtype=object)]:
            self.assertTrue(np.all(make_numeric(data) == data))

        for data in [
                np.array(['1', 2], dtype=object),
                pd.Series([1, '2', 3], dtype=object),
                pd.DataFrame([[1, '2.', 3], [4, '5.', 6]], dtype=object)]:
            self.assertTrue(np.all(data.astype(float) == make_numeric(data)))

        for data in [
                np.array(['1a', 2], dtype=object),
                pd.Series([1, '2a', 3], dtype=object),
                pd.DataFrame([[1, '2a.', 3], [4, '5a.', 6]], dtype=object)]:
            with self.assertRaises(ValueError):
                make_numeric(data)


if __name__ == '__main__':

    unittest.main() # pragma: no cover
