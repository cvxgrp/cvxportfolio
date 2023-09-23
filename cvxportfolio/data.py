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
"""This module include classes that download, store, and serve market data.

The two main abstractions are :class:`SymbolData` and :class:`MarketData`.
Neither are exposed outside this module. Their derived classes instead are.

If you want to implement the interface to another data source you should
derive from either of those two classes.

This module will be removed or heavily reformatted, most of it is
unused. The only parts that will remain are the FRED and YFinance
interfaces, simplified, and not meant to be accessed directly by users.
"""

import sqlite3
from pathlib import Path
import datetime
import warnings

import numpy as np
import pandas as pd
import requests

# from .estimator import DataEstimator

__all__ = ["YahooFinanceSymbolData", "FredSymbolData"]

BASE_LOCATION = Path.home() / "cvxportfolio_data"

class SymbolData:
    """Base class for a single symbol time series data.
    
    The data is either in the form of a Pandas Series or DataFrame
    and has datetime index.
    
    This class needs to be derived. At a minimum,
    one should redefine the ``_download`` method, which
    implements the downloading of the symbol's time series 
    from an external source. The method takes the current (already
    downloaded and stored) data and is supposed to **only append** to it.
    In this way we only store new data and don't modify already downloaded
    data.
    
    Additionally one can redefine the ``_preload`` method, which prepares
    data to serve to the user (so the data is stored in a different format
    than what the user sees.) We found that this separation can be useful.
    
    This class interacts with module-level functions named ``_load_BACKEND``
    and ``_store_BACKEND``, where ``BACKEND`` is the name of the storage
    system used. We define ``pickle``, ``csv``, and ``sqlite`` backends. 
    These may have limitations. See their docstrings for more information.
    
    
    :param symbol: the symbol that we download
    :type symbol: str
    :param storage_backend: the storage backend
    :type storage_backend: str
    :param base_storage_location: the location of the storage. We store in a 
        subdirectory named after the class which derives from this.
    :type base_storage_location: pathlib.Path
    """
    # """Base class for Cvxportfolio database interface.
    #
    # Provides a back-end independent way to load and store
    # pandas Series and DataFrames where the first index is a
    # pandas Timestamp (or numpy datetime64). It also provides
    # a systematic way to access external data sources via
    # the network. We specialize it to storing data locally
    # and downloading public time series of financial data.
    # By emulating some of these classes you can interface
    # cvxportfolio with other databases or other data sources.
    #
    # This interface is also used by cvxportfolio to store the
    # data it generates, such as Backtest classes data, or
    # estimators such as factor risk models.
    #
    # Cvxportfolio uses in-memory data whenever possible, and
    # in particular never uses BaseData methods during a backtest.
    # This ensures thread safety and allows us to use simple
    # local databases such as sqlite (or even flat csv files!).
    # Cvxportfolio loads data from the database before (possibly parallel)
    # backtesting, and stores it after backtesting. So, only one process
    # at a time accesses this class' methods. If you write custom callbacks
    # that are invoked during backtests, such as callables inside
    # DataEstimator, you should most probably not use cvxportfolio.data methods
    # inside them.
    #
    # LIMITATIONS:
    #     - columns names should be strings in order to work with all current
    #       data storage backends. If you create a DataFrame from a numpy array
    #       without specifying column names they will default to integers (0, 1, ...).
    #       If you store and load it back you will may get string column names ('0', '1', ...),
    #       depending on the backend.
    #     - the first level of the index should be a pandas Timestamp or equivalently
    #       numpy datetime64
    #     - you can only store sql-friendly data: integers, floats (including `np.nan`),
    #       datetime (including `np.datetime64('NaT')`), and simple alphanumeric strings
    #       (i.e., without commas or quote marks).
    #       If you need to store more complex python objects, such as the string
    #         "{'parameter1':3.0, 'parameter2': pd.Timestamp('2022-01-01')}",
    #       you will have to check that it works with the backend you use
    #      (it probably would not not with csv).
    # """
        
    def __init__(self, symbol, storage_backend, base_storage_location):
        self._symbol = symbol
        self._storage_backend = storage_backend
        self._base_storage_location = base_storage_location
        self._update()
        self._data = self.load()
    
    @property
    def storage_location(self):
        """Storage location. Directory is created if not existent.
        
        :rtype: pathlib.Path
        """
        loc = self._base_storage_location / f"{self.__class__.__name__}"
        loc.mkdir(parents=True, exist_ok=True)
        return loc
    
    @property
    def symbol(self):
        """The symbol whose data this instance contains.
        
        :rtype: str
        """
        return self._symbol
        
    @property
    def data(self):
        """Time series data, updated to the most recent observation.
        
        :rtype: pandas.Series or pandas.DataFrame
        """
        return self._data

    def _load_raw(self):
        """Load raw data from database."""
        # we could implement multiprocess safety here
        loader = globals()['_load_' + self._storage_backend]
        try:
            return loader(self.symbol, self.storage_location)
        except FileNotFoundError:
            return None

    def _load(self):
        """Load data from database using `self.preload` function to process.
        """
        return self._preload(self._load_raw())

    def _store(self, data):
        """Store data in database."""
        # we could implement multiprocess safety here
        storer = globals()['_store_' + self._storage_backend]
        storer(self.symbol, data, self.storage_location)

    def _update(self):
        """Update current stored data for symbol."""
        current = self._load_raw()
        updated = self._download(self.symbol, current)
        self._store(updated)
    
    def _download(self, symbol, current):
        """Download data from external source given already downloaded data.
        
        This method must be redefined by derived classes.
        
        :param symbol: The symbol we download.
        :type symbol: str
        :param current: The data already downloaded. We are supposed to
             **only append** to it. If None, no data is present.
        :type current: pandas.Series or pandas.DataFrame or None
        
        :rtype: pandas.Series or pandas.DataFrame
        
        """
        raise NotImplementedError

    def _preload(self, data):
        """Prepare data to serve to the user.
        
        This method can be redefined by derived classes.
        
        :param data: The data returned by the storage backend.
        :type data: pandas.Series or pandas.DataFrame
        
        :rtype: pandas.Series or pandas.DataFrame
        """
        return data


#
# Yahoo Finance.
#

class YahooFinanceSymbolData(SymbolData):
    """Yahoo Finance symbol data.
    
    
    """

    @staticmethod
    def _internal_process(data):
        """Manipulate yfinance data for better storing."""

        # nan-out nonpositive prices
        data.loc[data["Open"] <= 0, 'Open'] = np.nan
        data.loc[data["Close"] <= 0, "Close"] = np.nan
        data.loc[data["High"] <= 0, "High"] = np.nan
        data.loc[data["Low"] <= 0, "Low"] = np.nan
        data.loc[data["Adj Close"] <= 0, "Adj Close"] = np.nan

        # nan-out negative volumes
        data.loc[data["Volume"] < 0, 'Volume'] = np.nan

        intraday_logreturn = np.log(data["Close"]) - np.log(data["Open"])
        close_to_close_logreturn = np.log(data["Adj Close"]).diff().shift(-1)
        open_to_open_logreturn = (
            close_to_close_logreturn + intraday_logreturn -
            intraday_logreturn.shift(-1)
        )
        data["Return"] = np.exp(open_to_open_logreturn) - 1
        del data["Adj Close"]
        # eliminate last period's intraday data
        data.loc[data.index[-1], ["High", "Low",
                                  "Close", "Return", "Volume"]] = np.nan
        return data
    
    @staticmethod
    def _timestamp_convert(unix_seconds_ts):
        """Convert a UNIX timestamp in seconds to pd.Timestamp."""
        return pd.Timestamp(unix_seconds_ts*1E9, tz='UTC')
        
    @staticmethod
    def _now_timezoned():
        return pd.Timestamp(
            datetime.datetime.now(datetime.timezone.utc).astimezone())
    
    @staticmethod
    def _get_data_yahoo(ticker, start='1900-01-01', end='2100-01-01'):
        """Get 1 day OHLC from Yahoo finance. 
    
        Result is timestamped with the open time (time-zoned) of
        the instrument.
        """

        BASE_URL = 'https://query2.finance.yahoo.com'
    
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
            ' AppleWebKit/537.36 (KHTML, like Gecko)'
            ' Chrome/39.0.2171.95 Safari/537.36'}
        
        # print(HEADERS)
        start = int(pd.Timestamp(start).timestamp())
        end = int(pd.Timestamp(end).timestamp())
    
        res = requests.get(
            url=f"{BASE_URL}/v8/finance/chart/{ticker}", 
            params={'interval':'1d',
                "period1": start, 
                "period2": end}, 
            headers=HEADERS)
        
        # print(res)
        
        if res.status_code == 404:
            raise DataError(
                f'Data for symbol {ticker} is not available.'
                +'Json output:', str(res.json()))
    
        if res.status_code != 200:
            raise DataError(f'Yahoo finance data download failed. Json:',
                str(res.json()))
        
        data = res.json()['chart']['result'][0]

        index = pd.DatetimeIndex([
            YfinanceBase._timestamp_convert(el) 
            for el in data['timestamp']])
    
        df_result = pd.DataFrame(data['indicators']['quote'][0], index=index)
        df_result['adjclose'] = data['indicators']['adjclose'][0]['adjclose']
        
        # last timestamp is probably broken (not timed to market open)
        # we set its time to same as the day before, but this is wrong
        # on days of DST switch. It's fine though because that line will be
        # overwritten next update
        if df_result.index[-1].time() != df_result.index[-2].time():
            tm1 = df_result.index[-2].time()
            newlast = df_result.index[-1].replace(
                hour=tm1.hour, minute=tm1.minute, second=tm1.second)
            df_result.index = pd.DatetimeIndex(
                list(df_result.index[:-1]) + [newlast])

        # remove later, for now we match yfinance column names and ordering
        df_result = df_result[['open', 'high', 'low', 
                              'close', 'adjclose', 'volume']]
        df_result.columns = ['Open', 'High', 'Low', 
                             'Close', 'Adj Close', 'Volume']
        return df_result

    def _download(self, symbol, current=None, 
                overlap=5, grace_period='5d', **kwargs):
        """Download single stock from Yahoo Finance.

        If data was already downloaded we only download
        the most recent missing portion.

        Args:

            symbol (str): yahoo name of the instrument
            current (pandas.DataFrame or None): current data present locally
            overlap (int): how many lines of current data will be overwritten
                by newly downloaded data
            kwargs (dict): extra arguments passed to yfinance.download

        Returns:
            updated (pandas.DataFrame): updated DataFrame for the symbol
        """
        if overlap < 2:
            raise Exception(
                'There could be issues with DST and Yahoo finance data.')
        if (current is None) or (len(current) < overlap):
            updated = self._get_data_yahoo(symbol, **kwargs)
            return self._internal_process(updated)
        else:
            if (self._now_timezoned() - current.index[-1]
                ) < pd.Timedelta(grace_period):
                return current
            new = self._get_data_yahoo(symbol, start=current.index[-overlap])
            new = self._internal_process(new)
            return pd.concat([current.iloc[:-overlap], new])

    @staticmethod
    def _preload(data):
        """Prepare data for use by Cvxportfolio.

        We drop the 'Volume' column expressed in number of stocks and
        replace it with 'ValueVolume' which is an estimate of the (e.g.,
        US dollar) value of the volume exchanged on the day.
        """
        data["ValueVolume"] = data["Volume"] * data["Open"]
        del data["Volume"]
        # remove infty values
        data.iloc[:, :] = np.nan_to_num(
            data.values, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
        # remove extreme values
        # data.loc[data["Return"] < -.99, "Return"] = np.nan
        # data.loc[data["Return"] > .99, "Return"] = np.nan
        return data

#
# FRED.
#

class FredSymbolData(SymbolData):
    """Base class for FRED data access."""

    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    ## TODO: implement FRED point-in-time
    ## example:
    ## https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CES0500000003&vintage_date=2023-07-06
    ## hourly wages time series **as it appeared** on 2023-07-06 (try one day after, there's 1 new datapoint)
    ## store using pd.Series() of diff'ed values only, in practice they only revise recent 1-2 monthly
    ## obs.

    def _download(self, symbol):
        return pd.read_csv(self.URL + f'?id={symbol}', index_col=0, parse_dates=[0])[symbol]

    def download(self, symbol="DFF", current=None, grace_period='5d'):
        """Download or update pandas Series from FRED.

        If already downloaded don't change data stored locally and only
        add new entries at the end.

        Additionally, we allow for a `grace period`, if the data already
        downloaded has a last entry not older than the grace period, we
        don't download new data.
        """
        if current is None:
            return self._download(symbol)
        else:
            if (pd.Timestamp.today() - current.index[-1]) < pd.Timedelta(grace_period):
                return current

            new = self._download(symbol)
            new = new.loc[new.index > current.index[-1]]

            if new.empty:
                return current

            assert new.index[0] > current.index[-1]
            return pd.concat([current, new])


#
# Sqlite storage backend.
#

def _open_sqlite(storage_location):
    return sqlite3.connect(storage_location/"db.sqlite")

def _close_sqlite(connection):
    connection.close()
    
def _loader_sqlite(symbol, storage_location):
    """Load data in sqlite format.
    
    Limitations: if your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.
    """
    try:
        connection = _open_sqlite(storage_location)
        dtypes = pd.read_sql_query(
            f"SELECT * FROM {symbol}___dtypes",
            connection, index_col="index",
            dtype={"index": "str", "0": "str"})
        
        parse_dates='index'
        my_dtypes = dict(dtypes["0"])

        tmp = pd.read_sql_query(
            f"SELECT * FROM {symbol}", connection,
            index_col="index", parse_dates=parse_dates, dtype=my_dtypes)

        _close_sqlite(connection)
        multiindex = []
        for col in tmp.columns:
            if col[:8] == "___level":
                multiindex.append(col)
            else:
                break
        if len(multiindex):
            multiindex = [tmp.index.name] + multiindex
            tmp = tmp.reset_index().set_index(multiindex)
        return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp
    except pd.errors.DatabaseError:
        return None
            
def _storer_sqlite(symbol, data, storage_location):
    """Store data in sqlite format.

    We separately store dtypes for data consistency and safety.

    Limitations: if your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.

    """
    connection = _open_sqlite(storage_location)
    exists = pd.read_sql_query(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'",
        connection,
    )
    if len(exists):
        res = connection.cursor().execute(f"DROP TABLE '{symbol}'")
        res = connection.cursor().execute(
            f"DROP TABLE '{symbol}___dtypes'")
        connection.commit()

    if hasattr(data.index, "levels"):
        data.index = data.index.set_names(
            ["index"] +
            [f"___level{i}" for i in range(1, len(data.index.levels))]
        )
        data = data.reset_index().set_index("index")
    else:
        data.index.name = "index"
    
    if data.index[0].tzinfo is None:
        warnings.warn('Index has not timezone, setting to UTC')
        data.index = data.index.tz_localize('UTC')

    data.to_sql(f"{symbol}", connection)
    pd.DataFrame(data).dtypes.astype("string").to_sql(
        f"{symbol}___dtypes", connection)
    _close_sqlite(connection)


#
# Pickle storage backend.
#

def _loader_pickle(symbol, storage_location):
    """Load data in pickle format."""
    return pd.read_pickle(storage_location / f"{symbol}.pickle")

def _storer_pickle(symbol, data, storage_location):
    """Store data in pickle format."""
    data.to_pickle(storage_location / f"{symbol}.pickle")

#
# Csv storage backend.
#

def _loader_csv(symbol, storage_location):
    """Load data in csv format."""

    index_dtypes = pd.read_csv(
        storage_location / f"{symbol}___index_dtypes.csv",
        index_col=0)["0"]

    dtypes = pd.read_csv(
        storage_location / f"{symbol}___dtypes.csv", index_col=0,
        dtype={"index": "str", "0": "str"})
    dtypes = dict(dtypes["0"])
    new_dtypes = {}
    parse_dates = []
    for i, level in enumerate(index_dtypes):
        if "datetime64[ns" in level: # includes all timezones
            parse_dates.append(i)
    for i, el in enumerate(dtypes):
        if "datetime64[ns" in dtypes[el]:  # includes all timezones
            parse_dates += [i + len(index_dtypes)]
        else:
            new_dtypes[el] = dtypes[el]

    tmp = pd.read_csv(storage_location / f"{symbol}.csv",
        index_col=list(range(len(index_dtypes))),
        parse_dates=parse_dates, dtype=new_dtypes)

    return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp


def _storer_csv(symbol, data, storage_location):
    """Store data in csv format."""
    pd.DataFrame(data.index.dtypes if hasattr(data.index, 'levels')
        else [data.index.dtype]).astype("string").to_csv(
        storage_location / f"{symbol}___index_dtypes.csv")
    pd.DataFrame(data).dtypes.astype("string").to_csv(
        storage_location / f"{symbol}___dtypes.csv")
    data.to_csv(storage_location / f"{symbol}.csv")





# class RateBase(BaseData):
#     """Manipulate rate data from percent annualized to daily."""
#
#     trading_days = 252
#
#     def preload(self, data):
#         return np.exp(np.log(1 + data / 100) / self.trading_days) - 1
#
#
# class Yfinance(YfinanceBase, LocalDataStore):
#     """Yahoo Finance data interface using local data store.
#
#     Args:
#         base_location (pathlib.Path): filesystem directory where to store files.
#     """
#
#     def update_and_load(self, symbol):
#         """Update data for symbol and load it."""
#         return super().update_and_load(symbol)
#
#
# class FredRate(FredBase, RateBase, PickleStore):
#     """Load and store FRED rates like DFF."""
#
#     pass
#
#
# class YfinanceTimeSeries(DataEstimator, YfinanceBase, PickleStore):
#
#     def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
#         self.symbol = symbol
#         self.base_location = base_location
#         self.use_last_available_time = use_last_available_time
#
#     def _recursive_pre_evaluation(self, *args, **kwargs):
#         self.data = self.update_and_load(self.symbol)
#
#
# class FredTimeSeries(DataEstimator, FredBase, PickleStore):
#
#     def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
#         self.symbol = symbol
#         self.base_location = base_location
#         self.use_last_available_time = use_last_available_time
#
#     def _recursive_pre_evaluation(self, *args, **kwargs):
#         self.data = self.update_and_load(self.symbol)
#
#
# # class FredRateTimeSeries(DataEstimator, FredBase, RateBase, PickleStore):
# #
# #     def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
# #         self.symbol = symbol
# #         self.base_location = base_location
# #         self.use_last_available_time = use_last_available_time
# #
# #     def _recursive_pre_evaluation(self, *args, **kwargs):
# #         self.data = self.update_and_load(self.symbol)
#
#
# class TimeSeries(DataEstimator):
#     """Class for time series data managed by Cvxportfolio.
#
#     Args:
#         symbol (str): name of the time series, such as 'AAPL',
#             '^VIX', or 'DFF'.
#         source (str or BaseData): data source to use. Currently we
#             support 'yahoo', equivalent to `cvxportfolio.YfinanceBase`,
#             and 'fred', equivalent to `cvxportfolio.FredBase`. If you
#             implement your own you should define the `download` and
#             optionally `preload` methods. Default is 'yahoo'.
#         storage (str or BaseData): storage backend to use. Currently we
#             support 'sqlite', equivalent to `cvxportfolio.SqliteDataStore`,
#             and 'csv', equivalent to `cvxportfolio.LocalDataStore`. If you
#             implement your own you should define the `store` and
#             `load_raw` methods. Default is 'sqlite'.
#         use_last_available_time (bool): as in `cvxportfolio.DataEstimator`
#         base_location (pathlib.Path or None): base location for the data storage.
#     """
#
#     def __init__(
#         self,
#         symbol,
#         source="yahoo",
#         storage="pickle",
#         use_last_available_time=False,
#         base_location=None,
#     ):
#         self.symbol = symbol
#         if isinstance(source, str) and source == "yahoo":
#             source = YfinanceBase
#         if isinstance(source, str) and source == "fred":
#             source = FredRate
#
#         # from
#         # https://stackoverflow.com/questions/11042424/adding-base-class-to-existing-object-in-python
#         cls = self.__class__
#         self.__class__ = cls.__class__(
#             cls.__name__ + "With" + source.__name__, (cls, source), {}
#         )
#
#         # if isinstance(storage, str) and storage == "sqlite":
#         #     storage = SqliteDataStore
#         if isinstance(storage, str) and storage == "csv":
#             storage = LocalDataStore
#         if isinstance(storage, str) and storage == "pickle":
#             storage = PickleStore
#
#         cls = self.__class__
#         self.__class__ = cls.__class__(
#             cls.__name__ + "With" + storage.__name__, (cls, storage), {}
#         )
#
#         self.base_location = base_location
#         self.use_last_available_time = use_last_available_time
#         self.universe_maybe_noncash = None # fix, but we should retire this class
#
#     def _recursive_pre_evaluation(self, *args, **kwargs):
#         self.data = self.update_and_load(self.symbol)
