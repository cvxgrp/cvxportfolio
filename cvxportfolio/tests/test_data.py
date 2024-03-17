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

import socket
import sys
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.data import (DownloadedMarketData, Fred,
                               UserProvidedMarketData, YahooFinance,
                               _loader_csv, _loader_pickle, _loader_sqlite,
                               _storer_csv, _storer_pickle, _storer_sqlite)
from cvxportfolio.errors import DataError
from cvxportfolio.tests import CvxportfolioTest


class NoInternet():
    """Context with no internet.

    Adapted from: https://github.com/dvl/python-internet-sabotage
    """
    _module = sys.modules[__name__]

    def __init__(self, exception=IOError):
        self.exception = exception

    def _enable_socket(self):
        """Enable sockets in this module."""
        setattr(self._module, '_socket_disabled', False)

    def _disable_socket(self):
        """Disable sockets in this module."""
        setattr(self._module, '_socket_disabled', True)

        def _guarded(*args, **kwargs):
            """Monkey-patch the socket module."""
            if getattr(self._module, '_socket_disabled', False):
                raise self.exception('Internet is disabled')

            return socket.SocketType(*args, **kwargs)

        socket.socket = _guarded

    def __enter__(self):
        """Open context."""
        self._disable_socket()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close context, discard exceptions."""
        self._enable_socket()


class TestData(CvxportfolioTest):
    """Test SymbolData methods and interface."""

    def test_yfinance_download(self):
        """Test YfinanceBase."""

        storer = YahooFinance('AAPL', base_location=self.datadir)

        # pylint: disable=protected-access
        data = storer._download("AAPL", start="2023-04-01", end="2023-04-15")

        self.assertTrue(np.isclose(
            data.loc["2023-04-10 13:30:00+00:00", "return"],
            data.loc["2023-04-11 13:30:00+00:00", "open"] /
            data.loc["2023-04-10 13:30:00+00:00", "open"] - 1,
            rtol=1e-04, atol=1e-07,
        ))

    def test_fred(self):
        """Test basic Fred usage."""

        store = Fred(
            symbol="DFF", storage_backend='pickle',
            base_location=self.datadir)

        print(store.data)
        data = store.data
        self.assertTrue(np.isclose(data["2023-04-10"], 4.83))
        self.assertTrue(data.index[0] ==
            pd.Timestamp("1954-07-01 00:00:00+00:00"))

        # test update
        olddata = pd.Series(data.iloc[:-123], copy=True)
        olddata.index = olddata.index.tz_localize(None)
        # pylint: disable=protected-access
        newdata = store._preload(store._download("DFF", olddata))
        self.assertTrue(np.all(store.data == newdata))

        # test not re-downloading
        _ = Fred(
            symbol="DFF", grace_period=pd.Timedelta('10d'),
            base_location=self.datadir)

    def test_yahoo_finance(self):
        """Test yahoo finance ability to store and retrieve."""

        store = YahooFinance(
            symbol="ZM", storage_backend='pickle',
            base_location=self.datadir)

        data = store.data

        # print(data)

        self.assertTrue(np.isclose(
            data.loc["2023-04-05 13:30:00+00:00", "return"],
            data.loc["2023-04-06 13:30:00+00:00", "open"] /
            data.loc["2023-04-05 13:30:00+00:00", "open"] - 1,
            rtol=1e-04, atol=1e-07,
        ))

        store.update(grace_period=pd.Timedelta('1d'))
        data1 = store.load()
        # print(data1)

        self.assertTrue(np.isnan(data1.iloc[-1]["close"]))

        # print((data1.iloc[: len(data) - 1].Return -
        #       data.iloc[:-1].Return).describe().T)

        self.assertTrue(np.allclose(
            data1.loc[data.index[:-1]]['return'], data.iloc[:-1]['return']))

        # test not re-downloading
        _ = YahooFinance(
            symbol="ZM", grace_period=pd.Timedelta('10d'),
            base_location=self.datadir)

    def test_yahoo_finance_removefirstline(self):
        """Test that the first line of OHLCV is removed if there are NaNs."""

        # this symbol was found to have NaNs in the first line
        _ = YahooFinance(
            symbol="CVX", storage_backend='pickle',
            base_location=self.datadir)

    @unittest.skipIf(sys.version_info.major == 3
        and sys.version_info.minor < 11, "Issues with timezoned timestamps.")
    def test_sqlite3_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        with self.assertWarns(UserWarning):
            self._base_test_series(_loader_sqlite, _storer_sqlite)

    @unittest.skipIf(sys.version_info.major == 3
        and sys.version_info.minor < 11, "Issues with timezoned timestamps.")
    def test_local_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self._base_test_series(_loader_csv, _storer_csv)

    def test_pickle_store_series(self):
        """Test storing and retrieving of a Series with datetime index."""
        self._base_test_series(_loader_pickle, _storer_pickle)

    def test_sqlite3_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_dataframe(_loader_sqlite, _storer_sqlite)

    def test_local_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_dataframe(_loader_csv, _storer_csv)

    def test_pickle_store_dataframe(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_dataframe(_loader_pickle, _storer_pickle)

    def test_local_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_multiindex(_loader_csv, _storer_csv)

    def test_sqlite3_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_multiindex(_loader_sqlite, _storer_sqlite)

    def test_pickle_store_multiindex(self):
        """Test storing and retrieving of a DataFrame with datetime index."""
        self._base_test_multiindex(_loader_pickle, _storer_pickle)

    def _base_test_series(self, loader, storer):
        """Test storing and retrieving of a Series with datetime index."""

        for data in [
            pd.Series(
                0.0, pd.date_range("2020-01-01", "2020-01-10"),
                name="test0"),
            pd.Series(
                0.0, pd.date_range("2020-01-01", "2020-01-10", tz='UTC-05:00'),
                name="test1"),
            pd.Series(
                3, pd.date_range("2020-01-01", "2020-01-10", tz='UTC'),
                name="test2"),
            pd.Series("hello",
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC-05:00',
                    freq="h"),
                name="test3"),
            # test overwrite
            pd.Series("hello",
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC', freq="h"),
                name="test3"),
            # test datetime conversion
            pd.Series(
                pd.date_range("2022-01-01", "2022-01-02",  tz='UTC',
                    freq="h"),
                pd.date_range("2020-01-01", "2020-01-02",  tz='UTC', freq="h"),
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

    def _base_test_dataframe(self, loader, storer):
        """Test storing and retrieving of a DataFrame with datetime index."""

        index = pd.date_range("2020-01-01", "2020-01-02", freq="h", tz='UTC')
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

    def _base_test_multiindex(self, loader, storer):
        """Test storing and retrieving of a Series or DataFrame with multi-.

        index.
        """
        # second level is object
        timeindex = pd.date_range("2022-01-01", "2022-01-30", tz='UTC')
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
        timeindex = pd.date_range("2022-01-01", "2022-01-30", tz='UTC')
        second_level = pd.date_range("2022-01-01", "2022-01-03", tz='UTC')
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

    def test_download_errors(self):
        """Test single-symbol download error."""

        storer = YahooFinance(
            'AAPL', grace_period=self.data_grace_period,
            base_location=self.datadir)
        # with self.assertRaises(SyntaxError):
        #     # pylint: disable=protected-access
        #     storer._download('AAPL', overlap=1)

        class YahooFinanceErroneous(YahooFinance):
            """Modified YF that nans last open price."""
            def _download(
                    self, symbol, current=None, grace_period='5d', **kwargs):
                """Modified download method."""
                res = super()._download(symbol, current,
                    grace_period=grace_period)
                res.iloc[-1, 0 ] = np.nan
                return res

        _ = YahooFinanceErroneous('AMZN', base_location=self.datadir)
        with self.assertLogs(level='WARNING') as _:
            YahooFinanceErroneous(
                'AMZN', base_location=self.datadir)
            self.assertTrue(np.any([
                'changed last open price' in el for el in _.output]))

        class YahooFinanceErroneous2(YahooFinance):
            """Modified YF that nans some line."""
            def _download(
                    self, symbol, current=None, grace_period='5d', **kwargs):
                """Modified download method."""
                res = super()._download(symbol, current,
                    grace_period=grace_period)
                res.iloc[-20] = np.nan
                return res
        with self.assertLogs(level='WARNING') as _:
            _ = YahooFinanceErroneous2('GOOGL',
                base_location=self.datadir)
        with self.assertLogs(level='WARNING') as _:
            _ = YahooFinanceErroneous2(
                'GOOGL', base_location=self.datadir)

        class FredErroneous(Fred):
            """Modified FRED SymbolData that gives a NaN in the last entry."""

            def _download(self, symbol, current, grace_period):
                """Modified download method."""
                res = super()._download(symbol, current,
                    grace_period=grace_period)
                res.iloc[-1] = np.nan
                return res

        _ = FredErroneous('DFF', base_location=self.datadir)
        with self.assertLogs(level='WARNING') as _:
            FredErroneous(
                'DFF', base_location=self.datadir)
            self.assertTrue(np.any([
                'changed last value' in el for el in _.output]))

        class YahooFinanceErroneous3(YahooFinance):
            """Modified YF that is not append-only."""
            counter = 0
            def _download(
                    self, symbol, current=None, grace_period='5d', **kwargs):
                """Modified download method."""
                res = super()._download(symbol, current,
                    grace_period=grace_period)
                if self.counter > 0:
                    res.iloc[-2] = 0.
                self.counter += 1
                return res
        storer = YahooFinanceErroneous3('GOOGL', base_location=self.datadir)
        with self.assertLogs(level='WARNING') as _:
            storer.update(pd.Timedelta('0d'))

    def test_no_internet(self):
        """Test errors thrown when not connected to the internet."""

        with NoInternet():
            with self.assertRaises(DataError):
                cvx.YahooFinance('BABA', base_location=self.datadir)

        with NoInternet():
            with self.assertRaises(DataError):
                cvx.Fred('CES0500000003', base_location=self.datadir)

    def test_yahoo_finance_errors(self):
        """Test simple errors with Yahoo Finance."""

        with self.assertRaises(DataError):
            YahooFinance("DOESNTEXIST", base_location=self.datadir)

    def test_yahoo_finance_update(self):
        """Test specific issues when updating already stored data."""
        # pylint: disable=protected-access

        raw_data = pd.DataFrame(
            # skip last day because there might actually be issues
            # that invalidate the tests assumptions
            cvx.YahooFinance._get_data_yahoo('AAPL'), copy=True).iloc[:-1]

        # make sure last open is different from close of day before
        # (we use it later)
        if raw_data.iloc[-1, 0] == raw_data.iloc[-2, 3]:
            raw_data.iloc[-1, 0] += 0.1 # pragma: no cover

        # also do the same for day before...
        if raw_data.iloc[-2, 0] == raw_data.iloc[-3, 3]:
            raw_data.iloc[-2, 0] += 0.1 # pragma: no cover

        class YahooFinanceUpdaterTest(cvx.YahooFinance):
            """Tester of issues with update."""

            def _get_data_yahoo(
                    self, symbol, start=None, *args, **kwargs):
                return pd.DataFrame(
                    self.mock_data if start is None else
                    self.mock_data.loc[self.mock_data.index >= start], copy=True)

            @classmethod
            def _set_mock_data(cls, mock_data):
                cls.mock_data = mock_data

            def _delete_recent_stored(self, how_many):
                raw_stored = self._load_raw()
                only_past = pd.DataFrame(
                    raw_stored.iloc[:-how_many], copy=True)
                only_past.iloc[-1, -1] = np.nan
                self._store(only_past)

        YahooFinanceUpdaterTest._set_mock_data(raw_data)

        obj = YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # warm-up
        initial = YahooFinanceUpdaterTest(
            'AAPL', base_location=self.datadir).data
        updated = YahooFinanceUpdaterTest(
            'AAPL', base_location=self.datadir,
            grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, updated, equal_nan=True))
        obj._delete_recent_stored(10)
        re_updated = YahooFinanceUpdaterTest(
            'AAPL', base_location=self.datadir,
            grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))
        print(initial)

        # change intraday data keeping it valid, nothing happens
        raw_data_intraday_changed = pd.DataFrame(raw_data, copy=True)
        raw_data_intraday_changed.iloc[-1, 1] *= .9 # low
        raw_data_intraday_changed.iloc[-1, 2] *= 1.1 # high
        raw_data_intraday_changed.iloc[-1, 3] *= 1.05 # close
        raw_data_intraday_changed.iloc[-1, 4] *= 1.05 # adjclose
        YahooFinanceUpdaterTest._set_mock_data(raw_data_intraday_changed)
        re_updated = YahooFinanceUpdaterTest(
            'AAPL', base_location=self.datadir,
            grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # invalidate last open, gets filled w/ last close
        # this will only work as long as last open is different from
        # last close to begin with! that's why we edit it above
        raw_data_open_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_open_invalid.iloc[-1, 0] = np.inf
        YahooFinanceUpdaterTest._set_mock_data(raw_data_open_invalid)
        with self.assertLogs(level='WARNING') as _:
            re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
            self.assertTrue(np.any(
                ['changed last open price' in el for el in _.output]))
        # print(re_updated)
        self.assertTrue(re_updated.iloc[-1, 0] == re_updated.iloc[-2, 3])

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # make 3 days ago open invalid, nothing changes
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-3, 0] *= 100
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # make 3 days ago high invalid, nothing changes
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-3, 2] /= 100
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # make 2 days ago open invalid, stuff changes
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-2, 0] *= 100
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        with self.assertLogs(level='WARNING') as _:
            re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
            self.assertTrue(np.any(
                ['not append-only' in el for el in _.output]))
        # print(re_updated)
        self.assertTrue(re_updated.iloc[-2, 0] == re_updated.iloc[-3, 3])
        # self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # make 3 days ago high invalid, nothing changes
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-3, 2] *= 1000
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # make 2 days ago close invalid, stuff changes
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-2, 3] *= 1000
        # important: also adjclose needs to wrong in the same way,
        # this is generally true with Yahoo finance
        raw_data_recent_invalid.iloc[-2, 4] *= 1000
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        with self.assertLogs(level='INFO') as _:
            re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
            self.assertTrue(np.any(
                ['not append-only' in el for el in _.output]))
            # for el in _.output:
            #     print(el)
        # print(re_updated)
        self.assertTrue(re_updated.iloc[-2, 0] == re_updated.iloc[-1, 0])
        self.assertFalse(np.allclose(initial, re_updated, equal_nan=True))

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # worse still, invalidate first adjclose of recent data, no ffill
        # data will be eliminated and only last 3 rows will available
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-5, 4] *= 100
        raw_data_recent_invalid.iloc[-5, 3] *= 100
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        with self.assertLogs(level='INFO') as _:
            re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
            # for el in _.output:
            #     print(el)
            self.assertTrue(np.any(
                ['is eliminating data' in el for el in _.output]))

        # print(re_updated)
        self.assertTrue(np.allclose(initial, re_updated, equal_nan=True))

        # reset to init
        obj._delete_recent_stored(20)
        YahooFinanceUpdaterTest._set_mock_data(raw_data)
        YahooFinanceUpdaterTest('AAPL', base_location=self.datadir)

        # even worse, invalidate adjclose 3 and 4 days ago, no ffill
        # data will be eliminated and won't be able to concatenate
        # also change close otherwise dividends check kicks in
        raw_data_recent_invalid = pd.DataFrame(raw_data, copy=True)
        raw_data_recent_invalid.iloc[-3, 4] *= 100
        raw_data_recent_invalid.iloc[-4, 4] *= 100
        raw_data_recent_invalid.iloc[-3, 3] *= 100
        raw_data_recent_invalid.iloc[-4, 3] *= 100
        YahooFinanceUpdaterTest._set_mock_data(
            raw_data_recent_invalid)
        with self.assertLogs(level='INFO') as _:
            re_updated = YahooFinanceUpdaterTest(
                'AAPL', base_location=self.datadir,
                grace_period=pd.Timedelta('0d')).data
            self.assertTrue(np.any(
                ['is eliminating data' in el for el in _.output]))
            self.assertTrue(np.any(
                ['re-downloading from the start' in el for el in _.output]))
            # for el in _.output:
            #     print(el)
        # print(re_updated)
        # last open hasn't changed (re-download is with same bad data, but
        # ffill will work on fresh re-download)
        self.assertTrue(initial.iloc[-1, 0] == re_updated.iloc[-1, 0])

    def test_yahoo_finance_cleaning(self):
        """Test our logic to clean Yahoo Finance data."""

        # this stock was found to have NaN issues
        data = YahooFinance("ENI.MI", base_location=self.datadir).data
        self.assertTrue((data.valuevolume == 0).sum() > 0)
        self.assertTrue(data.iloc[:-1].isnull().sum().sum() == 0)

        # this stock was found to have phony open/low/high prices
        data = YahooFinance('NWG.L', base_location=self.datadir).data
        self.assertGreater(data['return'].min(), -0.75)
        self.assertLess(data['return'].max(), 0.75)

        # this stock had some extreme returns but they were legitimate
        data = YahooFinance('GME', base_location=self.datadir).data
        self.assertGreater(data['return'].min(), -0.75)
        self.assertGreater(data['return'].max(), 3)

    def test_yahoo_finance_preload_warnings(self):
        """Test warnings on _preload if data has issues."""

        # pylint: disable=protected-access

        raw_data = YahooFinance._get_data_yahoo('ZM')
        empty_instance = YahooFinance.__new__(YahooFinance)
        empty_instance._symbol = 'ZM' # because the warnings use the symbol
        cleaned = empty_instance._process(raw_data, None)

        def _test_warning(
                data_transformation, part_of_message, level='WARNING'):
            """Test that warning is raised w/ message containing some word."""
            data = pd.DataFrame(cleaned, copy=True)
            exec(data_transformation) # pylint: disable=exec-used
            # print(data)
            with self.assertLogs(level=level) as _:
                empty_instance._preload(data)
                # print(_)
                self.assertTrue(part_of_message in _.output[0])

        # columns are: open low high close volume return

        # high unexpected return
        _test_warning(
            'data.iloc[300,-1] = 4',
            'dubious total open-to-open returns')

        # low unexpected return
        _test_warning(
            'data.iloc[300,-1] = -0.9',
            'dubious total open-to-open returns')

        # low unexpected open
        _test_warning(
            'data.iloc[300,0] = data.iloc[300,0]*0.1',
            'dubious open to close returns',
            level='INFO')

        # high unexpected open
        _test_warning(
            'data.iloc[300,0] = data.iloc[300,0]*5',
            'dubious open to close returns',
            level='INFO')

        # low unexpected low
        _test_warning(
            'data.iloc[300,1] = data.iloc[300,1]*0.1',
            'dubious open to low returns',
            level='INFO')

        # high unexpected high
        _test_warning(
            'data.iloc[300,2] = data.iloc[300,2]*5',
            'dubious open to high returns',
            level='INFO')

    def test_yahoo_finance_remove_on_many_bad_adjcloses(self):
        """Test remove old data when many adjcloses are invalid."""

        # this stock was found to have bad (negative) adjcloses for many
        # months at its start
        with self.assertLogs(level='INFO') as _:
            YahooFinance('BATS.L', base_location=self.datadir)
            self.assertTrue(np.any(
                'contiguous' in el.output for el in _))

    def test_adjcloses_logrets_removal(self):
        """Test method to remove adjcloses when its logrets are anomalous."""

        # this stock had anomalous price changes in the 70s
        with self.assertLogs(level='INFO') as _:
            d = YahooFinance("SMT.L", base_location=self.datadir).data
            self.assertTrue(np.any([
                    'anomalous adjclose prices' in el for el in _.output]))
            self.assertTrue(d['return'].max() < 2)

        # this stock was found to have phony adjcloses
        with self.assertLogs(level='INFO') as _:
            YahooFinance('BA.L', base_location=self.datadir)
            self.assertTrue(np.any([
                    'anomalous adjclose prices' in el for el in _.output]))

        with self.assertLogs(level='INFO') as _:
            YahooFinance('BA.L', base_location=self.datadir)
            self.assertFalse(np.any([
                    'anomalous adjclose prices' in el for el in _.output]))

    def test_yahoo_finance_cleaning_granular(self):
        """Test each step of cleaning."""

        # pylint: disable=protected-access
        raw_data = YahooFinance._get_data_yahoo('ZM')
        # print(raw_data)
        empty_instance = YahooFinance.__new__(YahooFinance)
        empty_instance._symbol = 'ZM' # because the warnings use the symbol

        def _test_warning(
                data_transformation, part_of_message, level='WARNING'):
            """Test that warning is raised w/ message containing some word."""
            data = pd.DataFrame(raw_data, copy=True)
            exec(data_transformation) # pylint: disable=exec-used
            with self.assertLogs(level=level) as _:
                _cleaned = empty_instance._process(data, None)
                self.assertTrue(
                    np.any([part_of_message in el for el in _.output]))
                # check all NaNs have been filled
                self.assertTrue(_cleaned.iloc[:-1].isnull().sum().sum() == 0)

        def _test_warning_update(
                data_transformation, part_of_message, level='WARNING'):
            """Test that warning is raised w/ message containing some word."""
            new_data = pd.DataFrame(raw_data.iloc[-20:], copy=True)
            saved_data = pd.DataFrame(raw_data.iloc[:-15], copy=True)
            # no need to make it precise
            saved_data['return'] = np.log(saved_data.adjclose).diff()
            del saved_data['adjclose']
            exec(data_transformation) # pylint: disable=exec-used
            with self.assertLogs(level=level) as _:
                _cleaned = empty_instance._process(new_data, saved_data)
                self.assertTrue(
                    np.any([part_of_message in el for el in _.output]))
                # check all NaNs have been filled
                self.assertTrue(_cleaned.iloc[:-1].isnull().sum().sum() == 0)

        # missing initial adjclose
        _test_warning(
            'data.iloc[0,-2] = np.nan',
            'adjclose price is missing', level='INFO')

        # infty
        _test_warning(
            'data.iloc[2,2] = np.inf',
            'infinity', level='INFO')

        # non-pos price
        _test_warning(
            'data.iloc[2,0] = -1',
            'non-positive open', level='INFO')
        _test_warning(
            'data.iloc[2,0] = 0',
            'non-positive open', level='INFO')
        _test_warning(
            'data.iloc[4,2] = 0',
            'non-positive high', level='INFO')

        # neg volume
        _test_warning(
            'data.iloc[2,-1] = -1',
            'negative volumes', level='INFO')

        # open lower low
        _test_warning(
            'data.iloc[1,0] = data.iloc[1,1]*.9',
            'open price lower than low price', level='INFO')

        # open higher high
        _test_warning(
            'data.iloc[1,0] = data.iloc[1,2]*1.1',
            'open price higher than high price', level='INFO')

        # low higher close
        _test_warning(
            'data.iloc[3,1] = data.iloc[3].close * 1.1',
            'low price higher than close price', level='INFO')

        # high lower close
        _test_warning( # had to fix it otherwise open cleaner kicks in
            'close = data.iloc[3].close;'
            'data.iloc[3,0] = close * .95;' # open
            'data.iloc[3,1] = close * .95;' # low
            'data.iloc[3,2] = close * .975', # high
            'high price lower than close price', level='INFO')

        # extreme low price
        _test_warning(
            'data.iloc[3,1] = data.iloc[3,1] * .01',
            'anomalous low price', level='INFO')
        _test_warning(
            'data.iloc[3,1] = data.iloc[3,1] * .02',
            'anomalous low price', level='INFO')
        _test_warning(
            'data.iloc[3,1] = data.iloc[3,1] * .05',
            'anomalous low price', level='INFO')
        _test_warning(
            'data.iloc[3,1] = data.iloc[3,1] * .1',
            'anomalous low price', level='INFO')
        _test_warning(
            'data.iloc[3,1] = data.iloc[3,1] * .2',
            'anomalous low price', level='INFO')
        _test_warning( # changed dtindex until found one that works
            'data.iloc[20,1] = data.iloc[20,1] * .5',
            'anomalous low price', level='INFO')

        # extreme high price
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 100',
            'anomalous high price', level='INFO')
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 50',
            'anomalous high price', level='INFO')
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 20',
            'anomalous high price', level='INFO')
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 10',
            'anomalous high price', level='INFO')
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 5',
            'anomalous high price', level='INFO')
        _test_warning(
            'data.iloc[3,2] = data.iloc[3,2] * 2',
            'anomalous high price', level='INFO')

        # extreme open price
        _test_warning(
            'data.iloc[3,0] = data.iloc[3,0] * 2;'
            + 'data.iloc[3,2] = data.iloc[3,0]',
            'anomalous open price', level='INFO')
        _test_warning(
            'data.iloc[20,0] = data.iloc[20,0] * 0.5;'
            + 'data.iloc[20,1] = data.iloc[20,0]',
            'anomalous open price', level='INFO')

        # extreme open update
        _test_warning_update(
            'new_data.iloc[-1,0] = new_data.iloc[-1,0] * 1.75;'
            + 'new_data.iloc[-1,2] = new_data.iloc[-1,0]',
            'anomalous open price', level='INFO')
        _test_warning_update(
            'new_data.iloc[-1,0] = new_data.iloc[-1,0] *  0.5;'
            + 'new_data.iloc[-1,1] = new_data.iloc[-1,0]',
            'anomalous open price', level='INFO')

    # def test_yahoo_finance_wrong_last_time(self):
    #     """Test that we correct last time if intraday."""
    #
    #     class YahooFinanceErroneous4(YahooFinance):
    #         """Modified YF that sets last time wrong."""
    #         counter = 0
    #
    #         @staticmethod
    #         def _get_data_yahoo(
    #             ticker, start='1900-01-01', end='2100-01-01'):
    #             """Modified download method."""
    #             res = YahooFinance._get_data_yahoo(
    #                 ticker, start=start, end=end)
    #             if self.counter > 0:
    #                 res.index = list(res.index)[:-1] + [
    #                     res.index[-1] - pd.Timedelta('3h')]
    #             self.counter += 1
    #             print(res)
    #             return res
    #
    #     storer = YahooFinanceErroneous4('GOOGL', base_location=self.datadir)
    #     print(storer.data)
    #     #storer.update(pd.Timedelta('0d'))
    #     #print(storer.data)


class TestMarketData(CvxportfolioTest):
    """Test MarketData methods and interface."""

    def test_market_data_downsample(self):
        """Test downsampling of market data."""
        md = DownloadedMarketData(
            ['AAPL', 'GOOG'], grace_period=self.data_grace_period,
            base_location=self.datadir)

        # TODO: better to rewrite this test
        self.strip_tz_and_hour(md)

        idx = md.returns.index

        # not doing annual because XXXX-01-01 is holiday
        freqs = ['weekly', 'monthly', 'quarterly']
        testdays = ['2023-05-01', '2023-05-01', '2022-04-01']
        periods = [['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04',
                    '2023-05-05'],
                   idx[(idx >= '2023-05-01') & (idx < '2023-06-01')],
                   idx[(idx >= '2022-04-01') & (idx < '2022-07-01')]]

        for i, freq in enumerate(freqs):

            new_md = deepcopy(md)

            # pylint: disable=protected-access
            new_md._downsample(freq)
            print(new_md.returns)
            self.assertTrue(np.isnan(new_md.returns.GOOG.iloc[0]))
            self.assertTrue(np.isnan(new_md.volumes.GOOG.iloc[0]))
            self.assertTrue(np.isnan(new_md.prices.GOOG.iloc[0]))

            if freq == 'weekly':
                print((new_md.returns.index.weekday < 2).mean())
                self.assertTrue(
                    (new_md.returns.index.weekday < 2).mean() > .95)

            if freq == 'monthly':
                print((new_md.returns.index.day < 5).mean())
                self.assertTrue((new_md.returns.index.day < 5).mean() > .95)

            self.assertTrue(
                all(md.prices.loc[testdays[i]] ==
                    new_md.prices.loc[testdays[i]]))
            self.assertTrue(np.allclose(
                md.volumes.loc[periods[i]].sum(),
                new_md.volumes.loc[testdays[i]]))
            self.assertTrue(np.allclose(
                (1 + md.returns.loc[periods[i]]).prod(),
                1 + new_md.returns.loc[testdays[i]]))

    def test_market_data_methods(self):
        """Test objects returned by serve method of MarketDataInMemory."""
        t = self.returns.index[10]
        past_returns, current_returns, past_volumes, current_volumes, \
            current_prices = self.market_data.serve(t)
        self.assertTrue(current_returns.name == t)
        self.assertTrue(current_volumes.name == t)
        self.assertTrue(current_prices.name == t)
        self.assertTrue(np.all(past_returns.index < t))
        self.assertTrue(np.all(past_volumes.index < t))

    def test_market_data_object_safety(self):
        """Test safety of internal objects of MarketDataInMemory."""
        t = self.returns.index[10]

        past_returns, current_returns, past_volumes, current_volumes, \
            current_prices = self.market_data.serve(t)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        with self.assertRaises(ValueError):
            past_returns.iloc[-2, -2] = 2.
        with self.assertRaises(ValueError):
            current_returns.iloc[-3] = 2.
        with self.assertRaises(ValueError):
            past_volumes.iloc[-1, -1] = 2.
        with self.assertRaises(ValueError):
            current_volumes.iloc[-3] = 2.
        with self.assertRaises(ValueError):
            current_prices.iloc[-3] = 2.

        obj2 = deepcopy(self.market_data)
        # pylint: disable=protected-access
        obj2._set_read_only()

        past_returns, _, past_volumes, _, current_prices = obj2.serve(t)

        with self.assertRaises(ValueError):
            current_prices.iloc[-1] = 2.

        current_prices.loc['BABA'] = 3.

        past_returns, _, past_volumes, _, current_prices = obj2.serve(t)

        self.assertFalse('BABA' in current_prices.index)

    def test_user_provided_market_data(self):
        """Test UserProvidedMarketData."""

        used_returns = self.returns.iloc[:, :-1]
        used_returns.index = used_returns.index.tz_localize('UTC')
        used_volumes = pd.DataFrame(self.volumes, copy=True)
        t = used_returns.index[20]
        used_volumes.index = used_volumes.index.tz_localize('UTC')
        used_prices = pd.DataFrame(self.prices, copy=True)
        used_prices.index = used_prices.index.tz_localize('UTC')

        _ = UserProvidedMarketData(
            returns=used_returns, volumes=used_volumes, prices=used_prices,
            cash_key='USDOLLAR', base_location=self.datadir,
            min_history=pd.Timedelta('0d'))

        # add "cash" column
        md = UserProvidedMarketData(
            returns=used_returns, cash_key = 'cash',
            min_history=pd.Timedelta('0d'))
        self.assertEqual(md.returns.columns[-1], 'cash')
        self.assertEqual(md.returns['cash'].iloc[20], 0.)
        self.assertEqual(len(md.returns.columns), len(used_returns.columns)+1)

        without_prices = UserProvidedMarketData(
            returns=used_returns, volumes=used_prices, cash_key='USDOLLAR',
            base_location=self.datadir, min_history=pd.Timedelta('0d'))
        _, _, past_volumes, _,  current_prices = \
            without_prices.serve(t)
        self.assertTrue(current_prices is None)

        without_volumes = UserProvidedMarketData(
            returns=used_returns, cash_key='USDOLLAR',
            base_location=self.datadir, min_history=pd.Timedelta('0d'))
        _, _, past_volumes, current_volumes, \
            current_prices = without_volumes.serve(t)

        self.assertTrue(past_volumes is None)
        self.assertTrue(current_volumes is None)

        with self.assertRaises(SyntaxError):
            UserProvidedMarketData(returns=self.returns, volumes=self.volumes,
                       prices=self.prices.iloc[:, :-1], cash_key='cash',
                       min_history=pd.Timedelta('0d'))

        with self.assertRaises(DataError):
            UserProvidedMarketData(returns=self.returns.iloc[:10],
            cash_key='cash', min_history=pd.Timedelta('50d'))

        with self.assertRaises(SyntaxError):
            UserProvidedMarketData(
                returns=self.returns,
                volumes=self.volumes.iloc[:, :-3],
                prices=self.prices, cash_key='cash',
                min_history=pd.Timedelta('0d'))

        with self.assertRaises(SyntaxError):
            used_prices = pd.DataFrame(
                self.prices, index=self.prices.index,
                columns=self.prices.columns[::-1])
            UserProvidedMarketData(returns=self.returns, volumes=self.volumes,
                       prices=used_prices, cash_key='cash',
                       min_history=pd.Timedelta('0d'))

        with self.assertRaises(SyntaxError):
            used_volumes = pd.DataFrame(
                self.volumes, index=self.volumes.index,
                columns=self.volumes.columns[::-1])
            UserProvidedMarketData(returns=self.returns, volumes=used_volumes,
                       prices=self.prices, cash_key='cash',
                       min_history=pd.Timedelta('0d'))

        with self.assertRaises(NotImplementedError):
            UserProvidedMarketData(returns=self.returns, volumes=used_volumes,
                prices=self.prices, cash_key='NOTSUPPORTED',
                min_history=pd.Timedelta('0d'))

        with self.assertRaises(ValueError):
            UserProvidedMarketData(returns=self.returns, volumes=used_volumes,
                prices=self.prices, cash_key='USDOLLAR',
                min_history=pd.Timedelta('0d'))

        md = UserProvidedMarketData(
            returns=self.returns, volumes=self.volumes,
            prices=self.prices, cash_key='cash',
            min_history=pd.Timedelta('60d'))

        # try to serve when there's not enough min_history
        with self.assertRaises(ValueError):
            md.serve(t=self.returns.index[20])

        # errors with universe_selection_in_time
        with self.assertRaises(ValueError):
            UserProvidedMarketData(
                returns=self.returns, volumes=self.volumes,
                prices=self.prices, cash_key='cash',
                min_history=pd.Timedelta('60d'),
                universe_selection_in_time='not a dataframe')

        with self.assertRaises(ValueError):
            UserProvidedMarketData(
                returns=self.returns, volumes=self.volumes,
                prices=self.prices, cash_key='cash',
                min_history=pd.Timedelta('60d'),
                # has also cash
                universe_selection_in_time=pd.DataFrame(self.returns))

    def test_market_data_full(self):
        """Test serve method of DownloadedMarketData."""

        md = DownloadedMarketData(['AAPL', 'ZM'], base_location=self.datadir)
        assert np.all(md.full_universe == ['AAPL', 'ZM', 'USDOLLAR'])

        t = md.returns.index[-40]

        _, _, past_volumes, _, current_prices = md.serve(t)
        self.assertFalse(past_volumes is None)
        self.assertFalse(current_prices is None)

        # test error on missing cash return
        returns = pd.DataFrame(self.returns, copy=True)
        returns.iloc[20, -1] = np.nan
        with self.assertRaises(ValueError):
            UserProvidedMarketData(
                returns=returns, volumes=self.volumes,
                prices=self.prices, cash_key='cash',
                min_history=pd.Timedelta('60d')).serve(returns.index[20])

    def test_signature(self):
        """Test partial-universe signature of MarketData."""

        md = DownloadedMarketData(
            ['AAPL', 'ZM'], grace_period=self.data_grace_period,
            base_location=self.datadir)

        sig1 = md.partial_universe_signature(md.full_universe)

        md = DownloadedMarketData(
            ['AAPL', 'ZM'], grace_period=self.data_grace_period,
            trading_frequency='monthly', base_location=self.datadir)

        sig2 = md.partial_universe_signature(md.full_universe)

        self.assertFalse(sig1 == sig2)

        md = DownloadedMarketData(['AAPL', 'ZM', 'GOOG'],
            grace_period=self.data_grace_period,
            trading_frequency='monthly',
            base_location=self.datadir)

        sig3 = md.partial_universe_signature(
            pd.Index(['AAPL', 'ZM', 'USDOLLAR']))

        self.assertTrue(sig3 == sig2)

        md = DownloadedMarketData(['WM2NS'],
            datasource='Fred',
            grace_period=self.data_grace_period,
            base_location=self.datadir)

        print(md.partial_universe_signature(md.full_universe))

if __name__ == '__main__':

    unittest.main(warnings='error') # pragma: no cover
