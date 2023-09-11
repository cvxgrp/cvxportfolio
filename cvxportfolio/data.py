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
"""DEPRECATED.

This module will be removed or heavily reformatted, most of it is unused.
The only parts that will remain are the FRED and YFinance interfaces, simplified,
and not meant to be accessed directly by users.
"""

from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3

from .estimator import DataEstimator

__all__ = ["YfinanceTimeSeries", "FredTimeSeries",
           # "FredRateTimeSeries", 
           "BASE_LOCATION"]

BASE_LOCATION = Path.home() / "cvxportfolio_data"


class BaseData:
    """Base class for Cvxportfolio database interface.

    Provides a back-end independent way to load and store
    pandas Series and DataFrames where the first index is a
    pandas Timestamp (or numpy datetime64). It also provides
    a systematic way to access external data sources via
    the network. We specialize it to storing data locally
    and downloading public time series of financial data.
    By emulating some of these classes you can interface
    cvxportfolio with other databases or other data sources.

    This interface is also used by cvxportfolio to store the
    data it generates, such as Backtest classes data, or
    estimators such as factor risk models.

    Cvxportfolio uses in-memory data whenever possible, and
    in particular never uses BaseData methods during a backtest.
    This ensures thread safety and allows us to use simple
    local databases such as sqlite (or even flat csv files!).
    Cvxportfolio loads data from the database before (possibly parallel)
    backtesting, and stores it after backtesting. So, only one process
    at a time accesses this class' methods. If you write custom callbacks
    that are invoked during backtests, such as callables inside
    DataEstimator, you should most probably not use cvxportfolio.data methods
    inside them.

    LIMITATIONS:
        - columns names should be strings in order to work with all current
          data storage backends. If you create a DataFrame from a numpy array
          without specifying column names they will default to integers (0, 1, ...).
          If you store and load it back you will may get string column names ('0', '1', ...),
          depending on the backend.
        - the first level of the index should be a pandas Timestamp or equivalently
          numpy datetime64
        - you can only store sql-friendly data: integers, floats (including `np.nan`),
          datetime (including `np.datetime64('NaT')`), and simple alphanumeric strings
          (i.e., without commas or quote marks).
          If you need to store more complex python objects, such as the string
            "{'parameter1':3.0, 'parameter2': pd.Timestamp('2022-01-01')}",
          you will have to check that it works with the backend you use
         (it probably would not not with csv).
    """

    def load_raw(self, symbol):
        """Load raw data from database."""
        raise NotImplementedError

    def load(self, symbol):
        """Load data from database using `self.preload` function to process it."""
        return self.preload(self.load_raw(symbol))

    def store(self, symbol, data):
        """Store data in database."""
        raise NotImplementedError

    def download(self, symbol, current):
        """Download data from external source."""
        raise NotImplementedError

    def update(self, symbol):
        """Update current stored data for symbol."""
        current = self.load_raw(symbol)
        updated = self.download(symbol, current)
        self.store(symbol, updated)

    def update_and_load(self, symbol):
        """Update current stored data for symbol and load it.

        DEPRECATED: update and load functionalities have been separated.
        """
        current = self.load_raw(symbol)
        updated = self.download(symbol, current)
        self.store(symbol, updated)
        # return self.preload(updated)
        return self.load(symbol)

    def preload(self, data):
        """Prepare data to serve to the user."""
        return data


class YfinanceBase(BaseData):
    """Base class for the Yahoo Finance interface.

    This should not be used directly unless you know what you're doing.
    """

    @staticmethod
    def internal_process(data):
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

    def download(self, symbol, current=None, overlap=5, grace_period='5d', **kwargs):
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
        if (current is None) or (len(current) < overlap):
            updated = yf.download(symbol, progress=False, **kwargs)
            return self.internal_process(updated)
        else:
            if (pd.Timestamp.today() - current.index[-1]) < pd.Timedelta(grace_period):
                return current
            new = yf.download(symbol, progress=False,
                              start=current.index[-overlap], **kwargs)
            new = self.internal_process(new)
            return pd.concat([current.iloc[:-overlap], new])

    @staticmethod
    def preload(data):
        """Prepare data for use by Cvxportfolio.

        We drop the 'Volume' column expressed in number of stocks
        and replace it with 'ValueVolume' which is an estimate
        of the (e.g., US dollar) value of the volume exchanged
        on the day.
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


class BaseDataStore(BaseData):
    """Base class for data storage systems.

    Attributes:
        base_location (str or None): location of storage.
    """

    base_location = None


# class SqliteDataStore(BaseDataStore):
#     """Local sqlite3 database using python standard library.
#
#     Args:
#         location (pathlib.Path or None): pathlib.Path base location of the databases
#             directory or, if None, use ":memory:" for storing in RAM instead. Default
#             is ~/cvxportfolio/
#     """
#
#     def __init__(self, base_location=BASE_LOCATION):
#         """Initialize sqlite connection and if necessary create database."""
#         self.base_location = base_location
#
#     def __open__(self):
#         """Open database connection."""
#         if self.base_location is None:
#             self.connection = sqlite3.connect(":memory:")
#         else:
#             self.connection = sqlite3.connect(
#                 (self.base_location / self.__class__.__name__).with_suffix(".sqlite"))
#
#     def __close__(self):
#         """Close database connection."""
#         self.connection.close()
#
#     def store(self, symbol, data):
#         """Store Pandas object to sqlite.
#
#         We separately store dtypes for data consistency and safety.
#
#         Limitations: if your pandas object's index has a name it will be lost,
#             the index is renamed 'index'.
#         """
#         self.__open__()
#         exists = pd.read_sql_query(
#             f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'",
#             self.connection,
#         )
#         if len(exists):
#             res = self.connection.cursor().execute(f"DROP TABLE '{symbol}'")
#             res = self.connection.cursor().execute(
#                 f"DROP TABLE '{symbol}___dtypes'")
#             self.connection.commit()
#
#         if hasattr(data.index, "levels"):
#             data.index = data.index.set_names(
#                 ["index"] +
#                 [f"___level{i}" for i in range(1, len(data.index.levels))]
#             )
#             data = data.reset_index().set_index("index")
#         else:
#             data.index.name = "index"
#
#         data.to_sql(f"{symbol}", self.connection)
#         pd.DataFrame(data).dtypes.astype("string").to_sql(
#             f"{symbol}___dtypes", self.connection
#         )
#         self.__close__()
#
#     def load_raw(self, symbol):
#         """Load Pandas object with datetime index from sqlite.
#
#         If data is not present in in the database, return None.
#         """
#         try:
#             self.__open__()
#             dtypes = pd.read_sql_query(
#                 f"SELECT * FROM {symbol}___dtypes",
#                 self.connection,
#                 index_col="index",
#                 dtype={"index": "str", "0": "str"},
#             )
#             tmp = pd.read_sql_query(
#                 f"SELECT * FROM {symbol}",
#                 self.connection,
#                 index_col="index",
#                 parse_dates="index",
#                 dtype=dict(dtypes["0"]),
#             )
#             self.__close__()
#             multiindex = []
#             for col in tmp.columns:
#                 if col[:8] == "___level":
#                     multiindex.append(col)
#                 else:
#                     break
#             if len(multiindex):
#                 multiindex = [tmp.index.name] + multiindex
#                 tmp = tmp.reset_index().set_index(multiindex)
#             return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp
#         except pd.errors.DatabaseError:
#             return None


class PickleStore(BaseDataStore):
    """Pickle data store for pandas Series and DataFrames.

    Args:
        base_location (pathlib.Path): filesystem directory where to store files.

    """

    # base_location = BASE_LOCATION

    @property
    def location(self):
        return self.base_location / self.__class__.__name__

    def __init__(self, base_location=BASE_LOCATION):
        self.base_location = base_location

    def __create_if_not_existent(self):
        if not self.location.is_dir():
            self.location.mkdir(parents=True)
            print(f"Created folder at {self.location}")

    def load_raw(self, symbol, **kwargs):
        """Load raw data from local store."""
        try:
            return pd.read_pickle(self.location / f"{symbol}.pickle")
        except FileNotFoundError:
            return None

    def store(self, symbol, data, **kwargs):
        """Store data locally."""
        self.__create_if_not_existent()
        data.to_pickle(self.location / f"{symbol}.pickle")


class LocalDataStore(BaseDataStore):
    """Local data store for pandas Series and DataFrames.

    Args:
        base_location (pathlib.Path): filesystem directory where to store files.

    """

    # base_location = Path.home() / "cvxportfolio"

    @property
    def location(self):
        return self.base_location / self.__class__.__name__

    def __init__(self, base_location=BASE_LOCATION):
        self.base_location = base_location

    def __create_if_not_existent(self):
        if not self.location.is_dir():
            self.location.mkdir(parents=True)
            print(f"Created folder at {self.location}")

    def load_raw(self, symbol, **kwargs):
        """Load raw data from local store."""
        try:
            try:
                multiindex_types = pd.read_csv(
                    self.location /
                    f"{symbol}___multiindex_dtypes.csv",
                    index_col=0)["0"]
            except FileNotFoundError:
                multiindex_types = ["datetime64[ns]"]
            dtypes = pd.read_csv(
                self.location / f"{symbol}___dtypes.csv",
                index_col=0,
                dtype={"index": "str", "0": "str"},
            )
            dtypes = dict(dtypes["0"])
            new_dtypes = {}
            parse_dates = []
            for i, level in enumerate(multiindex_types):
                if level == "datetime64[ns]":
                    parse_dates.append(i)
            for i, el in enumerate(dtypes):
                if dtypes[el] == "datetime64[ns]":
                    parse_dates += [i + len(multiindex_types)]
                else:
                    new_dtypes[el] = dtypes[el]

            # raise Exception
            tmp = pd.read_csv(
                self.location / f"{symbol}.csv",
                index_col=list(range(len(multiindex_types))),
                parse_dates=parse_dates,
                **kwargs,
                dtype=new_dtypes,
            )
            return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp
        except FileNotFoundError:
            return None

    def store(self, symbol, data, **kwargs):
        """Store data locally."""
        self.__create_if_not_existent()
        if hasattr(data.index, "levels"):
            pd.DataFrame(data.index.dtypes).astype("string").to_csv(
                self.location / f"{symbol}___multiindex_dtypes.csv"
            )
        pd.DataFrame(data).dtypes.astype("string").to_csv(
            self.location / f"{symbol}___dtypes.csv"
        )
        data.to_csv(self.location / f"{symbol}.csv", **kwargs)


class FredBase(BaseData):
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

        Additionally, we allow for a `grace period`, if the data already downloaded
        has a last entry not older than the grace period, we don't download new data.
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


class RateBase(BaseData):
    """Manipulate rate data from percent annualized to daily."""

    trading_days = 252

    def preload(self, data):
        return np.exp(np.log(1 + data / 100) / self.trading_days) - 1


class Yfinance(YfinanceBase, LocalDataStore):

    """Yahoo Finance data interface using local data store.

    Args:
        base_location (pathlib.Path): filesystem directory where to store files.
    """

    def update_and_load(self, symbol):
        """Update data for symbol and load it."""
        return super().update_and_load(symbol)


class FredRate(FredBase, RateBase, PickleStore):
    """Load and store FRED rates like DFF."""

    pass


class YfinanceTimeSeries(DataEstimator, YfinanceBase, PickleStore):

    def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
        self.symbol = symbol
        self.base_location = base_location
        self.use_last_available_time = use_last_available_time

    def _recursive_pre_evaluation(self, *args, **kwargs):
        self.data = self.update_and_load(self.symbol)


class FredTimeSeries(DataEstimator, FredBase, PickleStore):

    def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
        self.symbol = symbol
        self.base_location = base_location
        self.use_last_available_time = use_last_available_time

    def _recursive_pre_evaluation(self, *args, **kwargs):
        self.data = self.update_and_load(self.symbol)


# class FredRateTimeSeries(DataEstimator, FredBase, RateBase, PickleStore):
#
#     def __init__(self, symbol, use_last_available_time=False, base_location=BASE_LOCATION):
#         self.symbol = symbol
#         self.base_location = base_location
#         self.use_last_available_time = use_last_available_time
#
#     def _recursive_pre_evaluation(self, *args, **kwargs):
#         self.data = self.update_and_load(self.symbol)


class TimeSeries(DataEstimator):
    """Class for time series data managed by Cvxportfolio.

    Args:
        symbol (str): name of the time series, such as 'AAPL',
            '^VIX', or 'DFF'.
        source (str or BaseData): data source to use. Currently we
            support 'yahoo', equivalent to `cvxportfolio.YfinanceBase`,
            and 'fred', equivalent to `cvxportfolio.FredBase`. If you
            implement your own you should define the `download` and
            optionally `preload` methods. Default is 'yahoo'.
        storage (str or BaseData): storage backend to use. Currently we
            support 'sqlite', equivalent to `cvxportfolio.SqliteDataStore`,
            and 'csv', equivalent to `cvxportfolio.LocalDataStore`. If you
            implement your own you should define the `store` and
            `load_raw` methods. Default is 'sqlite'.
        use_last_available_time (bool): as in `cvxportfolio.DataEstimator`
        base_location (pathlib.Path or None): base location for the data storage.


    """

    def __init__(
        self,
        symbol,
        source="yahoo",
        storage="pickle",
        use_last_available_time=False,
        base_location=None,
    ):
        self.symbol = symbol
        if isinstance(source, str) and source == "yahoo":
            source = YfinanceBase
        if isinstance(source, str) and source == "fred":
            source = FredRate

        # from
        # https://stackoverflow.com/questions/11042424/adding-base-class-to-existing-object-in-python
        cls = self.__class__
        self.__class__ = cls.__class__(
            cls.__name__ + "With" + source.__name__, (cls, source), {}
        )

        # if isinstance(storage, str) and storage == "sqlite":
        #     storage = SqliteDataStore
        if isinstance(storage, str) and storage == "csv":
            storage = LocalDataStore
        if isinstance(storage, str) and storage == "pickle":
            storage = PickleStore

        cls = self.__class__
        self.__class__ = cls.__class__(
            cls.__name__ + "With" + storage.__name__, (cls, storage), {}
        )

        self.base_location = base_location
        self.use_last_available_time = use_last_available_time
        self.universe_maybe_noncash = None # fix, but we should retire this class

    def _recursive_pre_evaluation(self, *args, **kwargs):
        self.data = self.update_and_load(self.symbol)
