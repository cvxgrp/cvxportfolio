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
"""Unit tests for the data interfaces."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cvxportfolio.data import (
    _loader_pickle, _storer_pickle, 
    _loader_csv, _storer_csv,
    _loader_sqlite, _storer_sqlite,
    FredSymbolData, YahooFinanceSymbolData)


class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize data directory."""
        cls.datadir = Path(tempfile.mkdtemp())
        print('created', cls.datadir)

    @classmethod
    def tearDownClass(cls):
        """Remove data directory."""
        print('removing', cls.datadir)
        shutil.rmtree(cls.datadir)

    # def test_time_series(self):
    #     ts = TimeSeries("ZM", base_location=self.datadir)
    #     assert not hasattr(ts, "data")
    #     ts._recursive_pre_evaluation()
    #     assert np.all(
    #         ts._recursive_values_in_time(
    #             pd.Timestamp("2023-04-11 13:30:00+00:00"), "foo", bar=None)
    #         == ts.data.loc["2023-04-11 13:30:00+00:00"]
    #     )

    def test_yfinance_download(self):
        """Test YfinanceBase."""

        data = YahooFinanceSymbolData._download("AAPL", start="2023-04-01", 
                                       end="2023-04-15")
        # print(data)
        # print(data.loc["2023-04-10 13:30:00+00:00"]["Return"])
        # print(data.loc["2023-04-11 13:30:00+00:00", "Open"] /
        #       data.loc["2023-04-10 13:30:00+00:00", "Open"] - 1)
        self.assertTrue(np.isclose(
            data.loc["2023-04-10 13:30:00+00:00", "Return"],
            data.loc["2023-04-11 13:30:00+00:00", "Open"] /
            data.loc["2023-04-10 13:30:00+00:00", "Open"] - 1,
        ))
        self.assertTrue(np.isnan(data.iloc[-1]["Close"]))

    def test_fred(self):
        """Test basic FRED usage."""
        
        store = FredSymbolData(
            symbol="DFF", storage_backend='pickle',
            base_storage_location=self.datadir)
        
        print(store.data)
        data = store.data
        self.assertTrue(np.isclose(data["2023-04-10"], 4.83))
        self.assertTrue(data.index[0] == 
            pd.Timestamp("1954-07-01 00:00:00+00:00"))
        
        # test update
        olddata = pd.Series(data.iloc[:-123], copy=True)
        olddata.index = olddata.index.tz_localize(None)
        newdata = store._preload(store._download("DFF", olddata))
        self.assertTrue(np.all(store.data == newdata))

    def test_yahoo_finance(self):
        """Test yahoo finance ability to store and retrieve."""
        
        store = YahooFinanceSymbolData(
            symbol="ZM", storage_backend='pickle',
            base_storage_location=self.datadir)
        
        data = store.data
        
        # print(data)

        self.assertTrue(np.isclose(
            data.loc["2023-04-05 13:30:00+00:00", "Return"],
            data.loc["2023-04-06 13:30:00+00:00", "Open"] /
            data.loc["2023-04-05 13:30:00+00:00", "Open"] - 1,
        ))
        
        store._update()
        data1 = store._load()
        # print(data1)

        self.assertTrue(np.isnan(data1.iloc[-1]["Close"]))

        # print((data1.iloc[: len(data) - 1].Return -
        #       data.iloc[:-1].Return).describe().T)

        self.assertTrue(np.allclose(
            data1.loc[data.index[:-1]].Return, data.iloc[:-1].Return))

    def test_sqlite3_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self.base_test_series(_loader_sqlite, _storer_sqlite)

    def test_local_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self.base_test_series(_loader_csv, _storer_csv)

    def test_pickle_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self.base_test_series(_loader_pickle, _storer_pickle)

    def test_sqlite3_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_dataframe(_loader_sqlite, _storer_sqlite)

    def test_local_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_dataframe(_loader_csv, _storer_csv)

    def test_pickle_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_dataframe(_loader_pickle, _storer_pickle)

    def test_local_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_multiindex(_loader_csv, _storer_csv)

    def test_sqlite3_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_multiindex(_loader_sqlite, _storer_sqlite)

    def test_pickle_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_multiindex(_loader_pickle, _storer_pickle)

    def base_test_series(self, loader, storer):
        """Test storing and retrieving of a Series with datetime index."""

        for data in [
            pd.Series(
                0.0, pd.date_range("2020-01-01", "2020-01-10", tz='UTC-05:00'), 
                name="test1"),
            pd.Series(
                3, pd.date_range("2020-01-01","2020-01-10", tz='UTC'), 
                name="test2"),
            pd.Series("hello", 
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC-05:00', 
                    freq="H"), 
                name="test3"),
            # test overwrite
            pd.Series("hello", 
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC', freq="H"), 
                name="test3"),
            # test datetime conversion
            pd.Series(
                pd.date_range("2022-01-01", "2022-01-02",  tz='UTC', 
                    freq="H"),
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC', freq="H"),
                name="test4"),
            ]:

            # print(data)
            # print(data.index)
            # print(data.index[0])
            # print(data.index[0].tzinfo)
            # print(data.index.dtype)
            # print(data.dtypes)

            storer(data.name, data, self.datadir)

            data1 = loader(data.name, self.datadir)
            # print(data1)
            # print(data1.index)
            # print(data1.index[0])
            # print(data1.index[0].tzinfo)
            # print(data1.index.dtype)
            # print(data1.dtypes)

            self.assertTrue(data.name == data1.name)
            self.assertTrue(all(data == data1))
            self.assertTrue(all(data.index == data1.index))
            self.assertTrue(data.dtypes == data1.dtypes)
        
        # test load not existent
        try:
            self.assertTrue(loader('blahblah', self.datadir) is None)
        except FileNotFoundError:
            pass

    def base_test_dataframe(self, loader, storer):
        """Test storing and retrieving of a DataFrame with datetime index."""

        index = pd.date_range("2020-01-01", "2020-01-02", freq="H", tz='UTC')
        data = {
            "one": range(len(index)),
            "two": np.arange(len(index)) / 19.0,
            "three": ["hello"] * len(index),
            "four": [np.nan] * len(index),
        }

        data["two"][2] = np.nan
        data = pd.DataFrame(data, index=index)
        # print(data)
        # print(data.index.dtype)
        # print(data.dtypes)

        storer("example", data, self.datadir)
        data1 = loader("example", self.datadir)
        # print(data1)
        # print(data1.index.dtype)
        # print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.dtypes == data1.dtypes))

    def base_test_multiindex(self, loader, storer):
        """Test storing and retrieving of a Series or DataFrame with multi-.

        index.
        """
        # second level is object
        timeindex = pd.date_range("2022-01-01", "2022-01-30",tz='UTC')
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 3), index=index)
        data.columns = ["one", "two", "tre"]

        # print(data.index)
        # print(data)
        # print(data.index.dtype)
        # print(data.dtypes)

        storer("example", data, self.datadir)
        data1 = loader("example", self.datadir)
        
        # print(data1.index)
        # print(data1)
        # print(data1.index.dtype)
        # print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.index.dtypes == data1.index.dtypes))
        self.assertTrue(all(data.dtypes == data1.dtypes))

        # second level is timestamp
        timeindex = pd.date_range("2022-01-01", "2022-01-30",tz='UTC')
        second_level = pd.date_range("2022-01-01", "2022-01-03",tz='UTC')
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 3), index=index)
        data.columns = ["a", "b", "c"]

        #print(data.index)
        # print(data)
        # print(data.index.dtypes)
        # print(data.dtypes)

        storer("example", data, self.datadir)
        data1 = loader("example", self.datadir)

        #print(data1.index)
        # print(data1)
        # print(data1.index.dtypes)
        # print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.index.dtypes == data1.index.dtypes))
        self.assertTrue(all(data.dtypes == data1.dtypes))


if __name__ == '__main__':
    unittest.main()
