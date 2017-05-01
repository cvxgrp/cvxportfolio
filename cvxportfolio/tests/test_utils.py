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

import os
import pickle

import pandas as pd
import numpy as np

from cvxportfolio import time_matrix_locator, time_locator, null_checker
from .base_test import BaseTest

DATAFILE = os.path.dirname(__file__) + os.path.sep + 'sample_data.pickle'


class TestUtils(BaseTest):

    def setUp(self):
        with open(DATAFILE, 'rb') as f:
            self.returns, self.sigma, self.volume, self.a, self.b, self.s = \
                pickle.load(f)
        self.volume['cash'] = np.NaN

    def test_null_checker(self):
        """Test null check."""
        self.assertRaises(ValueError, null_checker, np.nan)
        self.assertIsNone(null_checker(1.))
        self.assertIsNone(null_checker(self.returns))
        self.returns.iloc[0, 0] = np.NaN
        self.assertRaises(ValueError, null_checker, self.returns)

    def test_time_locator(self):
        """Test time locator."""
        t = self.returns.index[10]
        len(self.returns.columns)
        self.assertTrue(np.allclose(self.returns.loc[t],
                                    time_locator(self.returns, t)))
        self.assertTrue(np.allclose(23,
                                    time_locator(23, t)))
        self.assertTrue(np.allclose(self.returns.loc[t],
                                    time_locator(self.returns.loc[t], t)))

    def test_matrix_locator(self):
        """Test matrix locator."""
        index = ['a', 'b', 'c']
        df = pd.DataFrame(index=index, columns=index,
                          data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pn = pd.Panel({'1': df, '2': df*2})
        self.assertTrue(np.allclose(df.values,
                                    time_matrix_locator(df, t=12)))
        self.assertTrue(np.allclose(df.values*2,
                                    time_matrix_locator(pn, t='2')))
