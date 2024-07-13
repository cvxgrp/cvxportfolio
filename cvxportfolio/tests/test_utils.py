# Copyright (C) 2017-2024 Enzo Busseti
# Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
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
        # print(re)
        b = np.array([[1, 2, 3]])
        re1 = repr_numpy_pandas(b)
        # print(re1)
        self.assertTrue(re == re1)

        with self.assertRaises(NotImplementedError):
            repr_numpy_pandas('Hello!')

        a = pd.Series([1, 2, 3], ['2020-01-01', '2021-01-01', '2022-01-01'])
        re = repr_numpy_pandas(a)
        # print(re)
        b = pd.Series(a.values, a.index)
        re1 = repr_numpy_pandas(b)
        # print(re1)
        self.assertTrue(re == re1)

        # multiindexed df
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        data.columns = ["one", "two", "tre", "quattro",
                        "cinque", "sei", "sette", "otto", "nove", "dieci"]

        re = repr_numpy_pandas(data)
        # print(re)

        data1 = pd.DataFrame(data.values,
                             index=data.index, columns=data.columns)

        re1 = repr_numpy_pandas(data1)
        # print(re1)
        self.assertTrue(re == re1)

    def test_make_numeric(self):
        """Test make_numeric function for user-provided data."""

        for data in [
                2., np.array([1, 2]), pd.Series([1, 2, 3]),
                pd.DataFrame([[1, 2., 3], [4, 5., 6]])]:
            self.assertTrue(make_numeric(data) is data)

        for data in [
                np.array([1, 2], dtype=object),
                pd.Series([1, 2, 3], dtype=object),
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
