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

import cvxpy as cp
import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.utils import *


class TestUtils(unittest.TestCase):

    def test_hasher(self):
        a = np.array([1, 2, 3])
        re = repr_numpy_pandas(a)
        print(re)
        b = np.array([[1, 2, 3]])
        re1 = repr_numpy_pandas(b)
        print(re1)
        self.assertTrue(re == re1)

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


if __name__ == '__main__':

    unittest.main()
