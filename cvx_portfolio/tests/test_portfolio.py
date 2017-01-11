"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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


from .base_test import BaseTest
import numpy as np
import copy
import pandas as pd
from ..portfolio import Portfolio


class TestPortfolio(BaseTest):

    def test_array(self):
        """Test portfolio using array."""
        p = Portfolio(np.ones(10)*1000, cash_key=9, benchmark=np.ones(10)*.1)
        self.assertEqual(p.v, 10000)
        self.assertEqual(p.c, 1000)
        self.assertItemsAlmostEqual(p.w, np.ones(10)*0.1)
        self.assertItemsAlmostEqual(p.active_weights, np.zeros(10))
        # because the benchmark is too short
        self.assertRaises(AssertionError, Portfolio, np.ones(10) * 1000, benchmark=np.ones(10) * .1)
        p = Portfolio(np.ones(10) * 1000)
        self.assertEquals(p.cash_key, 'cash')
        p = Portfolio(np.ones(10) * 1000)
        self.assertItemsAlmostEqual(p.benchmark.values, [0.]*10+[1.])

    def test_copy(self):
        p = Portfolio(np.ones(10) * 1000, cash_key=9, benchmark=np.ones(10) * .1)
        self.assertEquals(p.cash_key, 9)
        p1 = copy.copy(p)
        self.assertEquals(p1.cash_key, 9)
        self.assertItemsAlmostEqual(p1.benchmark.values, [.1] * 10)

    def test_series(self):
        """Test portfolio using pandas series."""
        tickers = ['aaa', 'bbb', 'ccc', 'ddd']
        cash = 'cash'
        p = Portfolio(pd.Series(index=tickers+[cash], data=1E3),
                      cash_key='cash',
                      benchmark=pd.Series(index=tickers+[cash],
                                          data=1./5))
        self.assertItemsAlmostEqual(p.h, [1000.] * 5)
        self.assertItemsAlmostEqual(p.active_weights, [0.]*5)
