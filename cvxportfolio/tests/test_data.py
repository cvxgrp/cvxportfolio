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

import unittest
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cvxportfolio.data import (
    YfinanceBase,
    LocalDataStore,
    Yfinance,
    FredBase,
    FredRate,
    # SqliteDataStore,
    TimeSeries,
    PickleStore,
)


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

    def test_time_series(self):
        ts = TimeSeries("ZM", base_location=self.datadir)
        assert not hasattr(ts, "data")
        ts._recursive_pre_evaluation()
        assert np.all(
            ts._recursive_values_in_time(
                pd.Timestamp("2023-04-11"), "foo", bar=None)
            == ts.data.loc["2023-04-11"]
        )

    def test_yfinance_download(self):
        """Test YfinanceBase."""

        data = YfinanceBase().download("AAPL", start="2023-04-01", end="2023-04-15")
        print(data)
        print(data.loc["2023-04-10"]["Return"])
        print(data.loc["2023-04-11", "Open"] /
              data.loc["2023-04-10", "Open"] - 1)
        self.assertTrue(np.isclose(
            data.loc["2023-04-10", "Return"],
            data.loc["2023-04-11", "Open"] /
            data.loc["2023-04-10", "Open"] - 1,
        ))
        self.assertTrue(np.isnan(data.iloc[-1]["Close"]))

    def test_fred(self):
        store = FredRate(self.datadir)
        data = store.update_and_load("DFF")
        print(data)
        self.assertTrue(np.isclose((1 + data["2023-04-10"]) **
                                   store.trading_days, 1 + 4.83 / 100))

    def test_fred_base(self):
        data = FredBase().download("DFF")
        self.assertTrue(data["2023-04-06"] == 4.83)
        self.assertTrue(data.index[0] == pd.Timestamp("1954-07-01"))
        print(data)
        olddata = data.iloc[:-123]
        newdata = FredBase().download("DFF", olddata)
        self.assertTrue(np.all(data == newdata))

    def test_yfinance(self):
        """Test yfinance ability to store and retrieve."""

        store = Yfinance(self.datadir)
        data = store.update_and_load("ZM")

        print(data)

        self.assertTrue(np.isclose(
            data.loc["2023-04-05", "Return"],
            data.loc["2023-04-06", "Open"] /
            data.loc["2023-04-05", "Open"] - 1,
        ))

        data1 = store.update_and_load("ZM")
        print(data1)

        self.assertTrue(np.isnan(data1.iloc[-1]["Close"]))

        print((data1.iloc[: len(data) - 1].Return -
              data.iloc[:-1].Return).describe().T)

        self.assertTrue(np.allclose(
            data1.loc[data.index[:-1]].Return, data.iloc[:-1].Return))

    # def test_sqlite3_store_series(self):
    #     """Test storing and retrieving of a Series with datetime index."""
    #     self.base_test_series(SqliteDataStore, self.datadir)

    def test_local_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self.base_test_series(LocalDataStore, self.datadir)

    def test_pickle_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self.base_test_series(PickleStore, self.datadir)

    # def test_sqlite3_store_dataframe(self):
    #     """Test storing and retrieving of a DataFrame with datetime index."""
    #     self.base_test_dataframe(SqliteDataStore, self.datadir)

    def test_local_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_dataframe(LocalDataStore, self.datadir)

    def test_pickle_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_dataframe(PickleStore, self.datadir)

    def test_local_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_multiindex(LocalDataStore, self.datadir)

    # def test_sqlite3_store_multiindex(self):
    #     """Test storing and retrieving of a DataFrame with datetime index."""
    #     self.base_test_multiindex(SqliteDataStore, self.datadir)

    def test_pickle_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self.base_test_multiindex(PickleStore, self.datadir)

    def base_test_series(self, storeclass, *args, **kwargs):
        """Test storing and retrieving of a Series with datetime index."""
        store = storeclass(*args, **kwargs)
        for data in [
            pd.Series(
                0.0,
                pd.date_range(
                    "2020-01-01",
                    "2020-01-10"),
                name="prova1"),
            pd.Series(3, pd.date_range("2020-01-01",
                      "2020-01-10"), name="prova2"),
            pd.Series(
                "ciao", pd.date_range("2020-01-01", "2020-01-02", freq="H"), name="prova3"
            ),
            # test datetime conversion
            pd.Series(
                pd.date_range("2022-01-01", "2022-01-02", freq="H"),
                pd.date_range("2020-01-01", "2020-01-02", freq="H"),
                name="prova4",
            ),
            # test overwrite
            pd.Series(
                pd.date_range("2023-01-01", "2023-01-02", freq="H"),
                pd.date_range("2020-01-01", "2020-01-02", freq="H"),
                name="prova4",
            ),
        ]:
            print(data)
            print(data.index.dtype)
            print(data.dtypes)

            store.store(data.name, data)

            data1 = store.load(data.name)
            print(data1)
            print(data1.index.dtype)
            print(data1.dtypes)

            self.assertTrue(data.name == data1.name)
            self.assertTrue(all(data == data1))
            self.assertTrue(all(data.index == data1.index))
            self.assertTrue(data.dtypes == data1.dtypes)

        self.assertTrue(store.load("blahblah") is None)

    def base_test_dataframe(self, storeclass, *args, **kwargs):
        """Test storing and retrieving of a DataFrame with datetime index."""
        store = storeclass(*args, **kwargs)
        index = pd.date_range("2020-01-01", "2020-01-02", freq="H")
        data = {
            "one": range(len(index)),
            "two": np.arange(len(index)) / 19.0,
            "three": ["hello"] * len(index),
            "four": [np.nan] * len(index),
        }

        data["two"][2] = np.nan
        data = pd.DataFrame(data, index=index)
        print(data)
        print(data.index.dtype)
        print(data.dtypes)

        store.store("example", data)
        data1 = store.load("example")
        print(data1)
        print(data1.index.dtype)
        print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.dtypes == data1.dtypes))

    def base_test_multiindex(self, storeclass, *args, **kwargs):
        """Test storing and retrieving of a Series or DataFrame with multi-index."""
        store = storeclass(*args, **kwargs)

        # second level is object
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = ["hello", "ciao", "hola"]
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        data.columns = ["one", "two", "tre", "quattro",
                        "cinque", "sei", "sette", "otto", "nove", "dieci"]

        print(data.index)
        print(data)
        print(data.index.dtype)
        print(data.dtypes)

        store.store("example", data)
        data1 = store.load("example")

        print(data1.index)
        print(data1)
        print(data1.index.dtype)
        print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.index.dtypes == data1.index.dtypes))
        self.assertTrue(all(data.dtypes == data1.dtypes))

        # second level is timestamp
        timeindex = pd.date_range("2022-01-01", "2022-01-30")
        second_level = pd.date_range("2022-01-01", "2022-01-03")
        index = pd.MultiIndex.from_product([timeindex, second_level])
        data = pd.DataFrame(np.random.randn(len(index), 10), index=index)
        data.columns = [
            "one",
            "two",
            "tre",
            "quattro",
            "cinque",
            "sei",
            "sette",
            "otto",
            "nove",
            "dieci",
        ]

        print(data.index)
        print(data)
        print(data.index.dtype)
        print(data.dtypes)

        store.store("example", data)
        data1 = store.load("example")

        print(data1.index)
        print(data1)
        print(data1.index.dtype)
        print(data1.dtypes)

        self.assertTrue(all(data == data1))
        self.assertTrue(all(data.index == data1.index))
        self.assertTrue(all(data.index.dtypes == data1.index.dtypes))
        self.assertTrue(all(data.dtypes == data1.dtypes))


if __name__ == '__main__':
    unittest.main()
