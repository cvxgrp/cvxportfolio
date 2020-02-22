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

import pandas as pd
import numpy as np

from cvxportfolio import values_in_time, null_checker
from .base_test import BaseTest

DIR = os.path.dirname(__file__) + os.path.sep


class TestUtils(BaseTest):

    def setUp(self):
        self.sigma = pd.read_csv(DIR + 'sigmas.csv',
                                 index_col=0, parse_dates=[0])
        self.returns = pd.read_csv(DIR + 'returns.csv',
                                   index_col=0, parse_dates=[0])
        self.volume = pd.read_csv(DIR + 'volumes.csv',
                                  index_col=0, parse_dates=[0])
        self.a, self.b, self.s = 0.0005, 1., 0.
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
                                    values_in_time(self.returns, t)))
        self.assertTrue(np.allclose(23,
                                    values_in_time(23, t)))
        self.assertTrue(np.allclose(self.returns.loc[t],
                                    values_in_time(self.returns.loc[t], t)))

    # TODO drop Panel support and change this

    # def test_matrix_locator(self):
    #     """Test matrix locator."""
    #     index = ['a', 'b', 'c']
    #     df = pd.DataFrame(index=index, columns=index,
    #                       data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #     pn = pd.Panel({'1': df, '2': df * 2})
    #     self.assertTrue(np.allclose(df.values,
    #                                 time_matrix_locator(df, t=12)))
    #     self.assertTrue(np.allclose(df.values * 2,
    #                                 time_matrix_locator(pn, t='2')))
