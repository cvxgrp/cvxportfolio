# Copyright 2023 Enzo Busseti
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

If you want to interface cvxportfolio with financial data source other
than the ones we provide, you should derive from either of those two classes.
"""

import datetime
import logging
import sqlite3
import sys
import warnings
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import requests
import requests.exceptions

from .errors import DataError
from .utils import (hash_, periods_per_year_from_datetime_index,
                    resample_returns)

__all__ = ["YahooFinance", "Fred",
           "UserProvidedMarketData", "DownloadedMarketData"]

logger = logging.getLogger(__name__)

BASE_LOCATION = Path.home() / "cvxportfolio_data"

def now_timezoned():
    """Return current timestamp with local timezone.

    :returns: Current timestamp with local timezone.
    :rtype: pandas.Timestamp
    """
    return pd.Timestamp(
        datetime.datetime.now(datetime.timezone.utc).astimezone())

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

    This class interacts with module-level functions named ``_loader_BACKEND``
    and ``_storer_BACKEND``, where ``BACKEND`` is the name of the storage
    system used. We define ``pickle``, ``csv``, and ``sqlite`` backends.
    These may have limitations. See their docstrings for more information.


    :param symbol: The symbol that we downloaded.
    :type symbol: str
    :param storage_backend: The storage backend, implemented ones are
        ``'pickle'``, ``'csv'``, and ``'sqlite'``. By default ``'pickle'``.
    :type storage_backend: str
    :param base_location: The location of the storage. We store in a
        subdirectory named after the class which derives from this. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_location: pathlib.Path
    :param grace_period: If the most recent observation in the data is less
        old than this we do not download new data. By default it's one day.
    :type grace_period: pandas.Timedelta

    :attribute data: The downloaded data for the symbol.
    """

    def __init__(self, symbol,
                 storage_backend='pickle',
                 base_location=BASE_LOCATION,
                 grace_period=pd.Timedelta('1d')):
        self._symbol = symbol
        self._storage_backend = storage_backend
        self._base_location = base_location
        self.update(grace_period)
        self._data = self.load()

    @property
    def storage_location(self):
        """Storage location. Directory is created if not existent.

        :rtype: pathlib.Path
        """
        loc = self._base_location / f"{self.__class__.__name__}"
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
        loader = globals()['_loader_' + self._storage_backend]
        try:
            logger.info(
                f"{self.__class__.__name__} is trying to load {self.symbol}"
                + f" with {self._storage_backend} backend"
                + f" from {self.storage_location}")
            return loader(self.symbol, self.storage_location)
        except FileNotFoundError:
            return None

    def load(self):
        """Load data from database using `self.preload` function to process.

        :returns: Loaded time-series data for the symbol.
        :rtype: pandas.Series or pandas.DataFrame
        """
        return self._preload(self._load_raw())

    def _store(self, data):
        """Store data in database.

        :param data: Time-series data to store.
        :type data: pandas.Series or pandas.DataFrame
        """
        # we could implement multiprocess safety here
        storer = globals()['_storer_' + self._storage_backend]
        logger.info(
            f"{self.__class__.__name__} is storing {self.symbol}"
            + f" with {self._storage_backend} backend"
            + f" in {self.storage_location}")
        storer(self.symbol, data, self.storage_location)

    def _print_difference(self, current, new):
        """Helper method to print difference if update is not append-only.

        This is temporary and will be re-factored.
        """
        print("TEMPORARY: Diff between overlap of downloaded and stored")
        print((new - current).dropna(how='all').tail(5))

    def update(self, grace_period):
        """Update current stored data for symbol.

        :param grace_period: If the time between now and the last value stored
            is less than this, we don't update the data already stored.
        :type grace_period: pandas.Timedelta
        """
        current = self._load_raw()
        logger.info(
            f"Downloading {self.symbol}"
            + f" from {self.__class__.__name__}")
        updated = self._download(
            self.symbol, current, grace_period=grace_period)

        if np.any(updated.iloc[:-1].isnull()):
            logger.warning(
              " cvxportfolio.%s('%s').data contains NaNs."
              + " You may want to inspect it. If you want, you can delete the"
              + " data file in %s to force re-download from the start.",
              self.__class__.__name__, self.symbol, self.storage_location)

        try:
            if current is not None:
                if not np.all(
                        # we use numpy.isclose because returns may be computed
                        # via logreturns and numerical errors can sift through
                        np.isclose(updated.loc[current.index[:-1]],
                            current.iloc[:-1], equal_nan=True,
                            rtol=1e-08, atol=1e-08)):
                    logger.error(f"{self.__class__.__name__} update"
                        + f" of {self.symbol} is not append-only!")
                    self._print_difference(current, updated)
                if hasattr(current, 'columns'):
                    # the first column is open price
                    if not current.iloc[-1, 0] == updated.loc[
                            current.index[-1]].iloc[0]:
                        logger.error(
                            f"{self.__class__.__name__} update "
                            + f" of {self.symbol} changed last open price!")
                        self._print_difference(current, updated)
                else:
                    if not current.iloc[-1] == updated.loc[current.index[-1]]:
                        logger.error(
                            f"{self.__class__.__name__} update"
                            + f" of {self.symbol} changed last value!")
                        self._print_difference(current, updated)
        except KeyError:
            logger.error("%s update of %s could not be checked for"
                + " append-only edits. Was there a DST change?",
                self.__class__.__name__, self.symbol)
        self._store(updated)

    def _download(self, symbol, current, grace_period, **kwargs):
        """Download data from external source given already downloaded data.

        This method must be redefined by derived classes.

        :param symbol: The symbol we download.
        :type symbol: str
        :param current: The data already downloaded. We are supposed to
            **only append** to it. If None, no data is present.
        :type current: pandas.Series or pandas.DataFrame or None
        :rtype: pandas.Series or pandas.DataFrame
        """
        raise NotImplementedError #pragma: no cover

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

def _timestamp_convert(unix_seconds_ts):
    """Convert a UNIX timestamp in seconds to a pandas.Timestamp."""
    return pd.Timestamp(unix_seconds_ts*1E9, tz='UTC')


class YahooFinance(SymbolData):
    """Yahoo Finance symbol data.

    :param symbol: The symbol that we downloaded.
    :type symbol: str
    :param storage_backend: The storage backend, implemented ones are
        ``'pickle'``, ``'csv'``, and ``'sqlite'``.
    :type storage_backend: str
    :param base_storage_location: The location of the storage. We store in a
        subdirectory named after the class which derives from this.
    :type base_storage_location: pathlib.Path
    :param grace_period: If the most recent observation in the data is less
        old than this we do not download new data.
    :type grace_period: pandas.Timedelta

    :attribute data: The downloaded, and cleaned, data for the symbol.
    :type data: pandas.DataFrame
    """

    # is open-high-low-close-volume-(total)return
    IS_OHLCVR = True

    @staticmethod
    def _clean(data):
        """Clean Yahoo Finance open-close-high-low-volume-adjclose data."""

        # print(data)
        # print(data.isnull().sum())

        # nan-out nonpositive prices
        data.loc[data["open"] <= 0, 'open'] = np.nan
        data.loc[data["close"] <= 0, "close"] = np.nan
        data.loc[data["high"] <= 0, "high"] = np.nan
        data.loc[data["low"] <= 0, "low"] = np.nan
        data.loc[data["adjclose"] <= 0, "adjclose"] = np.nan

        # nan-out negative volumes
        data.loc[data["volume"] < 0, 'volume'] = np.nan

        # all infinity values are nans
        data.iloc[:, :] = np.nan_to_num(
            data.values, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # print(data)
        # print(data.isnull().sum())

        # if low is not the lowest, set it to nan
        data['low'].loc[
            data['low'] > data[['open', 'high', 'close']].min(1)] = np.nan

        # if high is not the highest, set it to nan
        data['high'].loc[
            data['high'] < data[['open', 'high', 'close']].max(1)] = np.nan

        # print(data)
        # print(data.isnull().sum())

        #
        # fills
        #

        # fill volumes with zeros (safest choice)
        data['volume'] = data['volume'].fillna(0.)

        # fill close price with open price
        data['close'] = data['close'].fillna(data['open'])

        # fill open price with close from day(s) before
        # repeat as long as it helps (up to 1 year)
        for shifter in range(252):
            orig_missing_opens = data['open'].isnull().sum()
            data['open'] = data['open'].fillna(data['close'].shift(
                shifter+1))
            new_missing_opens = data['open'].isnull().sum()
            if orig_missing_opens == new_missing_opens:
                break
            logger.info(
                "Filled missing open prices with close from %s periods before",
                shifter+1)

        # fill close price with same day's open
        data['close'] = data['close'].fillna(data['open'])

        # fill high price with max
        data['high'] = data['high'].fillna(data[['open', 'close']].max(1))

        # fill low price with max
        data['low'] = data['low'].fillna(data[['open', 'close']].min(1))

        # print(data)
        # print(data.isnull().sum())

        #
        # Compute returns
        #

        # compute log of ratio between adjclose and close
        log_adjustment_ratio = np.log(data['adjclose'] / data['close'])

        # forward fill adjustment ratio
        log_adjustment_ratio = log_adjustment_ratio.ffill()

        # non-market log returns (dividends, splits)
        non_market_lr = log_adjustment_ratio.diff().shift(-1)

        # full open-to-open returns
        open_to_open = np.log(data["open"]).diff().shift(-1)
        data['return'] = np.exp(open_to_open + non_market_lr) - 1

        # print(data)
        # print(data.isnull().sum())

        # intraday_logreturn = np.log(data["close"]) - np.log(data["open"])
        # close_to_close_logreturn = np.log(data["adjclose"]).diff().shift(-1)
        # open_to_open_logreturn = (
        #     close_to_close_logreturn + intraday_logreturn -
        #     intraday_logreturn.shift(-1)
        # )
        # data["return"] = np.exp(open_to_open_logreturn) - 1
        del data["adjclose"]

        # eliminate last period's intraday data
        data.loc[data.index[-1],
            ["high", "low", "close", "return", "volume"]] = np.nan

        # print(data)
        # print(data.isnull().sum())

        return data

    @staticmethod
    def _get_data_yahoo(ticker, start='1900-01-01', end='2100-01-01'):
        """Get 1 day OHLC from Yahoo finance.

        Result is timestamped with the open time (time-zoned) of the
        instrument.
        """

        base_url = 'https://query2.finance.yahoo.com'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
            ' AppleWebKit/537.36 (KHTML, like Gecko)'
            ' Chrome/39.0.2171.95 Safari/537.36'}

        # print(HEADERS)
        start = int(pd.Timestamp(start).timestamp())
        end = int(pd.Timestamp(end).timestamp())

        try:
            res = requests.get(
                url=f"{base_url}/v8/finance/chart/{ticker}",
                params={'interval': '1d',
                    "period1": start,
                    "period2": end},
                headers=headers,
                timeout=10) # seconds
        except requests.ConnectionError as exc:
            raise DataError(
                f"Download of {ticker} from YahooFinance failed."
                + " Are you connected to the Internet?") from exc

        # print(res)

        if res.status_code == 404:
            raise DataError(
                f'Data for symbol {ticker} is not available.'
                + 'Json output:', str(res.json()))

        if res.status_code != 200:
            raise DataError(f'Yahoo finance download of {ticker} failed. Json:',
                str(res.json())) # pragma: no cover

        data = res.json()['chart']['result'][0]

        try:
            index = pd.DatetimeIndex(
                [_timestamp_convert(el) for el in data['timestamp']])

            df_result = pd.DataFrame(
                data['indicators']['quote'][0], index=index)
            df_result['adjclose'] = data[
                'indicators']['adjclose'][0]['adjclose']
        except KeyError:
            raise DataError(f'Yahoo finance download of {ticker} failed.'
                + ' Json:', str(res.json())) # pragma: no cover

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

        return df_result[
            ['open', 'low', 'high', 'close', 'adjclose', 'volume']]

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
            raise SyntaxError(
                f'{self.__class__.__name__} with overlap smaller than 2'
                + ' could have issues with DST.')
        if (current is None) or (len(current) < overlap):
            updated = self._get_data_yahoo(symbol, **kwargs)
            logger.info('Downloading from the start.')
            result = self._clean(updated)
            # we remove first row if it contains NaNs
            if np.any(result.iloc[0].isnull()):
                result = result.iloc[1:]
            return result
        if (now_timezoned() - current.index[-1]
                ) < pd.Timedelta(grace_period):
            logger.info(
                'Skipping download because stored data is recent enough.')
            return current
        new = self._get_data_yahoo(symbol, start=current.index[-overlap])
        new = self._clean(new)
        return pd.concat([current.iloc[:-overlap], new])

    def _preload(self, data):
        """Prepare data for use by Cvxportfolio.

        We drop the `volume` column expressed in number of stocks and
        replace it with `valuevolume` which is an estimate of the (e.g.,
        US dollar) value of the volume exchanged on the day.
        """
        data["valuevolume"] = data["volume"] * data["open"]
        del data["volume"]

        return data

#
# Fred.
#

class Fred(SymbolData):
    """Fred single-symbol data.

    :param symbol: The symbol that we downloaded.
    :type symbol: str
    :param storage_backend: The storage backend, implemented ones are
        ``'pickle'``, ``'csv'``, and ``'sqlite'``. By default ``'pickle'``.
    :type storage_backend: str
    :param base_storage_location: The location of the storage. We store in a
        subdirectory named after the class which derives from this. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_storage_location: pathlib.Path
    :param grace_period: If the most recent observation in the data is less
        old than this we do not download new data. By default it's one day.
    :type grace_period: pandas.Timedelta

    :attribute data: The downloaded data for the symbol.
    """

    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # TODO: implement Fred point-in-time
    # example:
    # https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CES0500000003&vintage_date=2023-07-06
    # hourly wages time series **as it appeared** on 2023-07-06
    # store using pd.Series() of diff'ed values only.

    def _internal_download(self, symbol):
        try:
            return pd.read_csv(
                self.URL + f'?id={symbol}',
                index_col=0, parse_dates=[0])[symbol]
        except URLError as exc:
            raise DataError(f"Download of {symbol}"
                + f" from {self.__class__.__name__} failed."
                + " Are you connected to the Internet?") from exc

    def _download(
        self, symbol="DFF", current=None, grace_period='5d', **kwargs):
        """Download or update pandas Series from Fred.

        If already downloaded don't change data stored locally and only
        add new entries at the end.

        Additionally, we allow for a `grace period`, if the data already
        downloaded has a last entry not older than the grace period, we
        don't download new data.
        """
        if current is None:
            return self._internal_download(symbol)
        if (pd.Timestamp.today() - current.index[-1]
            ) < pd.Timedelta(grace_period):
            logger.info(
                'Skipping download because stored data is recent enough.')
            return current

        new = self._internal_download(symbol)
        new = new.loc[new.index > current.index[-1]]

        if new.empty:
            logger.info('New downloaded data is empty!')
            return current

        assert new.index[0] > current.index[-1]
        return pd.concat([current, new])

    def _preload(self, data):
        """Add UTC timezone."""
        data.index = data.index.tz_localize('UTC')
        return data

#
# Sqlite storage backend.
#

def _open_sqlite(storage_location):
    return sqlite3.connect(storage_location/"db.sqlite")

def _close_sqlite(connection):
    connection.close()

def _loader_sqlite(symbol, storage_location):
    """Load data in sqlite format.

    We separately store dtypes for data consistency and safety.

    .. note:: If your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.
    """
    try:
        connection = _open_sqlite(storage_location)
        dtypes = pd.read_sql_query(
            f"SELECT * FROM {symbol}___dtypes",
            connection, index_col="index",
            dtype={"index": "str", "0": "str"})

        parse_dates = 'index'
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
        if len(multiindex) > 0:
            multiindex = [tmp.index.name] + multiindex
            tmp = tmp.reset_index().set_index(multiindex)
        return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp
    except pd.errors.DatabaseError:
        return None

def _storer_sqlite(symbol, data, storage_location):
    """Store data in sqlite format.

    We separately store dtypes for data consistency and safety.

    .. note:: If your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.
    """
    connection = _open_sqlite(storage_location)
    exists = pd.read_sql_query(
      f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'",
      connection)

    if len(exists):
        _ = connection.cursor().execute(f"DROP TABLE '{symbol}'")
        _ = connection.cursor().execute(f"DROP TABLE '{symbol}___dtypes'")
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

#
# Market Data
#

class MarketData:
    """Prepare, hold, and serve market data.

    :method serve: Serve data for policy and simulator at time :math:`t`.
    """

    def serve(self, t):
        """Serve data for policy and simulator at time :math:`t`.

        :param t: Trading time. It must be included in the timestamps returned
            by :meth:`trading_calendar`.
        :type t: pandas.Timestamp

        :returns: past_returns, current_returns, past_volumes, current_volumes,
            current_prices
        :rtype: (pandas.DataFrame, pandas.Series, pandas.DataFrame,
            pandas.Series, pandas.Series)
        """
        raise NotImplementedError # pragma: no cover

    # pylint: disable=redundant-returns-doc
    def trading_calendar(
        self, start_time=None, end_time=None, include_end=True):
        """Get trading calendar between times.

        :param start_time: Initial time of the trading calendar. Always
            inclusive if present. If None, use the first available time.
        :type start_time: pandas.Timestamp
        :param end_time: Final time of the trading calendar. If None,
            use the last available time.
        :type end_time: pandas.Timestamp
        :param include_end: Include end time.
        :type include_end: bool

        :returns: Trading calendar.
        :rtype: pandas.DatetimeIndex
        """
        raise NotImplementedError # pragma: no cover

    @property
    def periods_per_year(self):
        """Average trading periods per year.

        :rtype: int
        """
        raise NotImplementedError # pragma: no cover

    @property
    def full_universe(self): # pylint: disable=redundant-returns-doc
        """Full universe, which might not be available for trading.

        :returns: Full universe.
        :rtype: pandas.Index
        """
        raise NotImplementedError # pragma: no cover

    # pylint: disable=unused-argument, redundant-returns-doc
    def partial_universe_signature(self, partial_universe):
        """Unique signature of this instance with a partial universe.

        A partial universe is a subset of the full universe that is
        available at some time for trading.

        This is used in cvxportfolio.cache to sign back-test caches that
        are saved on disk. If not redefined it returns None which disables
        on-disk caching.

        :param partial_universe: A subset of the full universe.
        :type partial_universe: pandas.Index

        :returns: Signature.
        :rtype: str
        """
        return None

# compiled based on Interactive Brokers benchmark rates choices
# (see https://www.ibkrguides.com/kb/article-2949.htm)
# and their FRED codes
RATES = {
    'USDOLLAR': 'DFF', # Federal funds effective rate
    'EURO': 'ECBESTRVOLWGTTRMDMNRT', # BCE short term rate
    'GBPOUND': 'IUDSOIA', # SONIA
    'JPYEN': 'IRSTCB01JPM156N', # updated monthly
    }

class MarketDataInMemory(MarketData):
    """Market data that is stored in memory when initialized."""

    # this is overwritten in the derived classes' initializers
    returns = None

    def __init__(
        self, trading_frequency, base_location, cash_key, min_history,
        online_usage = False):
        """This must be called by the derived classes."""
        if (self.returns.index[-1] - self.returns.index[0]) < min_history:
            raise DataError(
                "The provided returns have less history "
                + f"than the min_history {min_history}")
        if trading_frequency:
            self._downsample(trading_frequency)
        self.trading_frequency = trading_frequency

        self._set_read_only()
        self._check_sizes()
        self._mask = None
        self._masked_returns = None
        self._masked_volumes = None
        self._masked_prices = None
        self.base_location = Path(base_location)
        self.cash_key = cash_key
        self._min_history_timedelta = min_history
        self.online_usage = online_usage

    def _mask_dataframes(self, mask):
        """Mask internal dataframes if necessary."""
        if (self._mask is None) or not np.all(self._mask == mask):
            logger.info("Masking internal %s dataframes.",
                self.__class__.__name__)
            colmask = self.returns.columns[mask]
            # self._masked_returns = self._df_or_ser_set_read_only(
            #     pd.DataFrame(self.returns.iloc[:, mask], copy=True))
            self._masked_returns = self._df_or_ser_set_read_only(
               pd.DataFrame(self.returns.loc[:, colmask], copy=True))
            # self._masked_returns = self._df_or_ser_set_read_only(
            #     pd.DataFrame(np.array(self.returns.values[:, mask]),
            #         index=self.returns.index, columns=colmask))
            if not self.volumes is None:
                # self._masked_volumes = self._df_or_ser_set_read_only(
                #     pd.DataFrame(self.volumes.iloc[:, mask[:-1]], copy=True))
                self._masked_volumes = self._df_or_ser_set_read_only(
                    pd.DataFrame(self.volumes.loc[:, colmask[:-1]], copy=True))
                # self._masked_volumes = self._df_or_ser_set_read_only(
                #     pd.DataFrame(np.array(self.volumes.values[:, mask[:-1]]),
                #         index=self.volumes.index, columns=colmask[:-1]))
            if not self.prices is None:
                # self._masked_prices = self._df_or_ser_set_read_only(
                #     pd.DataFrame(self.prices.iloc[:, mask[:-1]], copy=True))
                self._masked_prices = self._df_or_ser_set_read_only(
                    pd.DataFrame(self.prices.loc[:, colmask[:-1]], copy=True))
            self._mask = mask

    @property
    def full_universe(self):
        """Full universe, which might not be available for trading.

        :returns: Full universe.
        :rtype: pandas.Index
        """
        return self.returns.columns

    def serve(self, t):
        """Serve data for policy and simulator at time :math:`t`.

        :param t: Time of execution, *e.g.*, stock market open of a given day.
        :type t: pandas.Timestamp

        :returns: (past_returns, current_returns, past_volumes,
            current_volumes, current_prices)
        :rtype: (pandas.DataFrame, pandas.Series, pandas.DataFrame or None,
            pandas.Series or None, pandas.Series or None)
        """

        mask = self._universe_mask_at_time(t).values
        self._mask_dataframes(mask)

        tidx = self.returns.index.get_loc(t)
        past_returns = self._df_or_ser_set_read_only(
            pd.DataFrame(self._masked_returns.iloc[:tidx]))
        current_returns = self._df_or_ser_set_read_only(
            pd.Series(self._masked_returns.iloc[tidx]))

        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            past_volumes = self._df_or_ser_set_read_only(
                pd.DataFrame(self._masked_volumes.iloc[:tidx]))
            current_volumes = self._df_or_ser_set_read_only(
                pd.Series(self._masked_volumes.iloc[tidx]))
        else:
            past_volumes = None
            current_volumes = None

        if not self.prices is None:
            tidx = self.prices.index.get_loc(t)
            current_prices = self._df_or_ser_set_read_only(
                pd.Series(self._masked_prices.iloc[tidx]))
        else:
            current_prices = None

        return (past_returns, current_returns, past_volumes, current_volumes,
                current_prices)

    def _add_cash_column(self, cash_key, grace_period):
        """Add the cash column to an already formed returns dataframe.

        This assumes that the trading periods are about equally spaced.
        If, say, you have trading periods with very different lengths you
        should redefine this method **and** replace the :class:`CashReturn`
        objective term.
        """

        if not cash_key in RATES:
            raise NotImplementedError(
                'Currently the only data pipelines built are for cash_key'
                f' in {list(RATES)}')

        if self.returns.index.tz is None:
            raise DataError(
                'Your provided dataframes are not timezone aware.'
                + " This is not recommended, and doesn't allow to add the cash"
                + " returns' column internally."
                + " You can fix this by adding a timezone manually "
                + "using pandas.DataFrame.tz_localize to the dataframes before"
                + " you pass them, or you can provide"
                + " the cash returns' column as the last column of the returns"
                + " dataframe (so it has one more column than volumes and"
                + " prices, if provided), and set the cash_key parameter to"
                + " its name.")

        data = Fred(
            RATES[cash_key], base_location=self.base_location,
            grace_period=grace_period)

        cash_returns_per_period = resample_returns(
            data.data/100, periods=self.periods_per_year)

        # we merge instead of assigning column because indexes might
        # be misaligned (e.g., with tz-aware timestamps)
        cash_returns_per_period.name = self.cash_key
        original_returns_index = self.returns.index
        tmp = pd.concat([self.returns, cash_returns_per_period], axis=1)
        tmp[cash_key] = tmp[cash_key].ffill()
        self.returns = tmp.loc[original_returns_index]

    def trading_calendar(
        self, start_time=None, end_time=None, include_end=True):
        """Get trading calendar from market data.

        :param start_time: Initial time of the trading calendar. Always
            inclusive if present. If None, use the first available time.
        :type start_time: pandas.Timestamp
        :param end_time: Final time of the trading calendar. If None,
            use the last available time.
        :type end_time: pandas.Timestamp
        :param include_end: Include end time.
        :type include_end: bool

        :returns: Trading calendar.
        :rtype: pandas.DatetimeIndex
        """
        result = self.returns.index
        result = result[result >= self._earliest_backtest_start]
        if start_time:
            result = result[result >= start_time]
        if end_time:
            result = result[(result <= end_time)]
        if not include_end:
            result = result[:-1]
        return result

    def _universe_mask_at_time(self, t):
        """Return the valid universe mask at time t."""
        past_returns = self.returns.loc[self.returns.index < t]
        if self.online_usage:
            valid_universe_mask = past_returns.count() >= self.min_history
        else:
            valid_universe_mask = ((past_returns.count() >= self.min_history) &
                (~self.returns.loc[t].isnull()))
        if sum(valid_universe_mask) <= 1:
            raise DataError(
                f'The trading universe at time {t} has size less or equal'
                + ' than one, i.e., only the cash account. There are probably '
                + ' issues with missing data in the provided market returns.')
        return valid_universe_mask

    @staticmethod
    def _df_or_ser_set_read_only(df_or_ser):
        """Set numpy array contained in dataframe to read only.

        This is done on data store internally before it is served to the
        policy or the simulator to ensure data consistency in case some
        element of the pipeline accidentally corrupts the data.

        This is enough to prevent direct assignement to the resulting
        dataframe. However it could still be accidentally corrupted by
        assigning to columns or indices that are not present in the
        original. We avoid that case as well by returning a wrapped
        dataframe (which doesn't copy data on creation) in
        serve_data_policy and serve_data_simulator.
        """
        data = df_or_ser.values
        data.flags.writeable = False
        if hasattr(df_or_ser, 'columns'):
            return pd.DataFrame(data, index=df_or_ser.index,
                                columns=df_or_ser.columns)
        return pd.Series(data, index=df_or_ser.index, name=df_or_ser.name)

    def _set_read_only(self):
        """Set internal dataframes to read-only."""

        self.returns = self._df_or_ser_set_read_only(self.returns)

        if not self.prices is None:
            self.prices = self._df_or_ser_set_read_only(self.prices)

        if not self.volumes is None:
            self.volumes = self._df_or_ser_set_read_only(self.volumes)

    @property
    def _earliest_backtest_start(self):
        """Earliest date at which we can start a backtest."""
        return self.returns.iloc[:, :-1].dropna(how='all').index[
            self.min_history]

    sampling_intervals = {
        'weekly': 'W-MON', 'monthly': 'MS', 'quarterly': 'QS', 'annual': 'AS'}

    # @staticmethod
    # def _is_first_interval_small(datetimeindex):
    #     """Check if post-resampling the first interval is small.
    #
    #     We have no way of knowing exactly if the first interval
    #     needs to be dropped. We drop it if its length is smaller
    #     than the average of all others, minus 2 standard deviation.
    #     """
    #     first_interval = (datetimeindex[1] - datetimeindex[0])
    #     all_others = (datetimeindex[2:] - datetimeindex[1:-1])
    #     return first_interval < (all_others.mean() - 2 * all_others.std())

    def _downsample(self, interval):
        """_downsample market data."""
        if not interval in self.sampling_intervals:
            raise SyntaxError(
                'Unsopported trading interval for down-sampling.')
        interval = self.sampling_intervals[interval]
        new_returns_index = pd.Series(self.returns.index, self.returns.index
                                      ).resample(interval, closed='left',
                                                 label='left').first().values
        # print(new_returns_index)
        self.returns = np.exp(np.log(
            1+self.returns).resample(interval, closed='left', label='left'
                                     ).sum(min_count=1))-1
        self.returns.index = new_returns_index

        # last row is always unknown
        self.returns.iloc[-1] = np.nan

        # # we drop the first row if its interval is small
        # if self._is_first_interval_small(self.returns.index):
        #     self.returns = self.returns.iloc[1:]

        # we nan-out the first non-nan element of every col
        for col in self.returns.columns[:-1]:
            self.returns[col].loc[
                    (~(self.returns[col].isnull())).idxmax()
                ] = np.nan

        # and we drop the first row, which is mostly NaNs anyway
        self.returns = self.returns.iloc[1:]

        if self.volumes is not None:
            new_volumes_index = pd.Series(
                self.volumes.index, self.volumes.index
                    ).resample(interval, closed='left',
                               label='left').first().values
            self.volumes = self.volumes.resample(
                interval, closed='left', label='left').sum(min_count=1)
            self.volumes.index = new_volumes_index

            # last row is always unknown
            self.volumes.iloc[-1] = np.nan

            # # we drop the first row if its interval is small
            # if self._is_first_interval_small(self.volumes.index):
            #     self.volumes = self.volumes.iloc[1:]

            # we nan-out the first non-nan element of every col
            for col in self.volumes.columns:
                self.volumes[col].loc[
                        (~(self.volumes[col].isnull())).idxmax()
                    ] = np.nan

            # and we drop the first row, which is mostly NaNs anyway
            self.volumes = self.volumes.iloc[1:]

        if self.prices is not None:
            new_prices_index = pd.Series(
                self.prices.index, self.prices.index
                ).resample(
                    interval, closed='left', label='left').first().values
            self.prices = self.prices.resample(
                interval, closed='left', label='left').first()
            self.prices.index = new_prices_index

            # # we drop the first row if its interval is small
            # if self._is_first_interval_small(self.prices.index):
            #     self.prices = self.prices.iloc[1:]

            # we nan-out the first non-nan element of every col
            for col in self.prices.columns:
                self.prices[col].loc[
                        (~(self.prices[col].isnull())).idxmax()
                    ] = np.nan

            # and we drop the first row, which is mostly NaNs anyway
            self.prices = self.prices.iloc[1:]

    def _check_sizes(self):
        """Check sizes of user-provided dataframes."""

        if (not self.volumes is None) and (
                not (self.volumes.shape[1] == self.returns.shape[1] - 1)
                or not all(self.volumes.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Volumes should have same columns as returns, minus cash_key.')

        if (not self.prices is None) and (
                not (self.prices.shape[1] == self.returns.shape[1] - 1)
                or not all(self.prices.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Prices should have same columns as returns, minus cash_key.')

    @property
    def periods_per_year(self):
        """Average trading periods per year inferred from the data.

        :returns: Average periods per year.
        :rtype: int
        """
        return periods_per_year_from_datetime_index(self.returns.index)

    @property
    def min_history(self):
        """Min history expressed in periods.

        :returns: How many non-null elements of the past returns for a given
            name are required to include it.
        :rtype: int
        """
        return int(np.round(self.periods_per_year * (
            self._min_history_timedelta / pd.Timedelta('365.24d'))))


class UserProvidedMarketData(MarketDataInMemory):
    """User-provided market data.

    :param returns: Historical open-to-open returns. The return
        at time :math:`t` is :math:`r_t = p_{t+1}/p_t -1` where
        :math:`p_t` is the (open) price at time :math:`t`. Must
        have datetime index. You can also include cash
        returns as its last column, and set ``cash_key`` below to the last
        column's name.
    :type returns: pandas.DataFrame
    :param volumes: Historical market volumes, expressed in units
        of value (*e.g.*, US dollars).
    :type volumes: pandas.DataFrame or None
    :param prices: Historical open prices (*e.g.*, used for rounding
        trades in the :class:`MarketSimulator`).
    :type prices: pandas.DataFrame or None
    :param trading_frequency: Instead of using frequency implied by
        the index of the returns, down-sample all dataframes.
        We implement ``'weekly'``, ``'monthly'``, ``'quarterly'`` and
        ``'annual'``. By default (None) don't down-sample.
    :type trading_frequency: str or None
    :param min_history: Minimum amount of time for which the returns
         are not ``np.nan`` before each assets enters in a back-test.
    :type min_history: pandas.Timedelta
    :param base_location: The location of the storage, only used
        in case it downloads the cash returns. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_location: pathlib.Path
    :param cash_key: Name of the cash account. If not the last column
        of the provided returns, it will be downloaded. In that case you should
        make sure your provided dataframes have a timezone aware datetime
        index. Its returns are the risk-free rate.
    :type cash_key: str
    :param online_usage: Disable removal of assets that have ``np.nan`` returns
        for the given time. Default False.
    :type online_usage: bool
    """

    # pylint: disable=too-many-arguments
    def __init__(self, returns, volumes=None, prices=None,
                 copy_dataframes=True, trading_frequency=None,
                 min_history=pd.Timedelta('365.24d'),
                 base_location=BASE_LOCATION,
                 grace_period=pd.Timedelta('1d'),
                 cash_key='USDOLLAR',
                 online_usage=False):

        if returns is None:
            raise SyntaxError(
                "If you don't specify a universe you should pass `returns`.")

        self.base_location = Path(base_location)
        self.cash_key = cash_key

        self.returns = pd.DataFrame(returns, copy=copy_dataframes)
        self.volumes = volumes if volumes is None else\
            pd.DataFrame(volumes, copy=copy_dataframes)
        self.prices = prices if prices is None else\
            pd.DataFrame(prices, copy=copy_dataframes)

        if cash_key != returns.columns[-1]:
            self._add_cash_column(cash_key, grace_period=grace_period)

        # this is mandatory
        super().__init__(
            trading_frequency=trading_frequency,
            base_location=base_location,
            cash_key=cash_key,
            min_history=min_history,
            online_usage=online_usage)


class DownloadedMarketData(MarketDataInMemory):
    """Market data that is downloaded.

    :param universe: List of names as understood by the data source
        used, *e.g.*, ``['AAPL', 'GOOG']`` if using the default
        Yahoo Finance data source.
    :type universe: list
    :param datasource: The data source used.
    :type datasource: str or :class:`SymbolData` class
    :param cash_key: Name of the cash account, its rates will be downloaded
        and added as last columns of the returns. Its returns are the
        risk-free rate.
    :type cash_key: str
    :param base_location: The location of the storage. By default
        it's a directory named ``cvxportfolio_data`` in your home folder.
    :type base_location: pathlib.Path
    :param storage_backend: The storage backend, implemented ones are
        ``'pickle'``, ``'csv'``, and ``'sqlite'``. By default ``'pickle'``.
    :type storage_backend: str
    :param min_history: Minimum amount of time for which the returns
         are not ``np.nan`` before each assets enters in a back-test.
    :type min_history: pandas.Timedelta
    :param grace_period: If the most recent observation of each symbol's
        data is less old than this we do not download new data.
        By default it's one day.
    :type grace_period: pandas.Timedelta
    :param trading_frequency: Instead of using frequency implied by
        the index of the returns, down-sample all dataframes.
        We implement ``'weekly'``, ``'monthly'``, ``'quarterly'`` and
        ``'annual'``. By default (None) don't down-sample.
    :type trading_frequency: str or None
    :param online_usage: Disable removal of assets that have ``np.nan`` returns
        for the given time. Default False.
    :type online_usage: bool
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 universe=(),
                 datasource='YahooFinance',
                 cash_key='USDOLLAR',
                 base_location=BASE_LOCATION,
                 storage_backend='pickle',
                 min_history=pd.Timedelta('365.24d'),
                 grace_period=pd.Timedelta('1d'),
                 trading_frequency=None,
                 online_usage=False):
        """Initializer."""

        # drop duplicates and ensure ordering
        universe = sorted(set(universe))

        self.base_location = Path(base_location)
        self.cash_key = cash_key
        if isinstance(datasource, type):
            self.datasource = datasource
        else: # try to load in current module
            self.datasource = globals()[datasource]
        self._get_market_data(
            universe, grace_period=grace_period,
            storage_backend=storage_backend)
        self._add_cash_column(self.cash_key, grace_period=grace_period)
        self._remove_missing_recent()

        # this is mandatory
        super().__init__(
            trading_frequency=trading_frequency,
            base_location=base_location,
            cash_key=cash_key,
            min_history=min_history,
            online_usage=online_usage)

    def _get_market_data(self, universe, grace_period, storage_backend):
        """Download market data."""
        database_accesses = {}
        print('Updating data', end='')
        sys.stdout.flush()

        for stock in universe:
            logger.info(
                'Updating %s with %s.', stock, self.datasource.__name__)
            print('.', end='')
            sys.stdout.flush()
            database_accesses[stock] = self.datasource(
                stock, base_location=self.base_location,
                grace_period=grace_period, storage_backend=storage_backend)
        print()

        if hasattr(self.datasource, 'IS_OHLCVR') and self.datasource.IS_OHLCVR:
            self.returns = pd.DataFrame(
                {stock: database_accesses[stock].data['return']
                for stock in universe})
            self.volumes = pd.DataFrame(
                {stock: database_accesses[stock].data['valuevolume']
                for stock in universe})
            self.prices = pd.DataFrame(
                {stock: database_accesses[stock].data['open']
                for stock in universe})
        else:  # for now only Fred for indexes, we assume prices!
            assert isinstance(database_accesses[universe[0]].data, pd.Series)
            self.prices = pd.DataFrame(
                # open prices
                {stock: database_accesses[stock].data for stock in universe})
            self.returns = 1 - self.prices / self.prices.shift(-1)
            self.volumes = None

    def _remove_missing_recent(self):
        """Clean recent data.

        Yahoo Finance may has issues with most recent data; we remove
        recent days if there are NaNs.
        """

        if self.prices.iloc[-5:].isnull().any().any():
            logger.debug(
                'Removing some recent lines because there are missing values.')
            drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
            logger.debug('Dropping at index %s', drop_at)
            self.returns = self.returns.loc[self.returns.index < drop_at]
            if self.prices is not None:
                self.prices = self.prices.loc[self.prices.index < drop_at]
            if self.volumes is not None:
                self.volumes = self.volumes.loc[self.volumes.index < drop_at]

        # for consistency we must also nan-out the last row
        # of returns and volumes
        self.returns.iloc[-1] = np.nan
        if self.volumes is not None:
            self.volumes.iloc[-1] = np.nan

    def partial_universe_signature(self, partial_universe):
        """Unique signature of this instance with a partial universe.

        A partial universe is a subset of the full universe that is
        available at some time for trading.

        This is used in cvxportfolio.cache to sign back-test caches that
        are saved on disk. See its implementation below for details. If
        not redefined it returns None which disables on-disk caching.

        :param partial_universe: A subset of the full universe.
        :type partial_universe: pandas.Index

        :returns: Signature.
        :rtype: str
        """
        assert isinstance(partial_universe, pd.Index)
        assert np.all(partial_universe.isin(self.full_universe))
        result = f'{self.__class__.__name__}('
        result += f'datasource={self.datasource.__name__}, '
        result += f'partial_universe_hash={hash_(np.array(partial_universe))},'
        result += f' trading_frequency={self.trading_frequency})'
        return result
