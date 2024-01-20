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
"""This module defines the :class:`SymbolData` abstraction and derived classes."""
import datetime
import logging
import sqlite3
import warnings
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import requests
import requests.exceptions

from ..errors import DataError

logger = logging.getLogger(__name__)

BASE_LOCATION = Path.home() / "cvxportfolio_data"

__all__ = [
    '_loader_csv', '_loader_pickle', '_loader_sqlite',
    '_storer_csv', '_storer_pickle', '_storer_sqlite',
    'Fred', 'SymbolData', 'YahooFinance', 'BASE_LOCATION']

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

class OHLCV(SymbolData): # pylint: disable=abstract-method
    """Base class for Open-High-Low-Close-Volume symbol data."""

    # TODO: factor quality check and clean into total-return related and non-

class OHLCVTR(OHLCV): # pylint: disable=abstract-method
    """Base class for Open-High-Low-Close-Volume-Total Return symbol data."""

    # TODO: consider creating a OHLCVAC (adjusted closes) subclass

    # is open-high-low-close-volume-total return
    IS_OHLCVR = True

    # rolstd windows for finding wrong logreturns
    _ROLSTD_WINDOWS = [20, 60, 252]

    # threshold for finding wrong logreturns
    _WRONG_LOGRET_THRESHOLD = 15

    def _indexes_extreme_logrets_wrt_rolstddev(self, lrets, window, treshold):
        """Get indexes of logreturns that are extreme wrt trailing stddev."""
        trailing_stdev = np.sqrt((lrets**2).rolling(window).median().shift(1))
        bad_indexes = lrets.index[np.abs(lrets / trailing_stdev) > treshold]
        return bad_indexes

    def _find_wrong_daily_logreturns(self, lrets):
        """Find indexes of logreturns that are most probably data errors."""
        bad_indexes = []
        for window in self._ROLSTD_WINDOWS:
            bad_indexes.append(
                set(self._indexes_extreme_logrets_wrt_rolstddev(
                lrets, window=window, treshold=self._WRONG_LOGRET_THRESHOLD)))
            bad_indexes.append(
                set(self._indexes_extreme_logrets_wrt_rolstddev(
                lrets.iloc[::-1], window=window,
                treshold=self._WRONG_LOGRET_THRESHOLD)))
        bad_indexes = set.intersection(*bad_indexes)
        return bad_indexes

    # TODO: plan
    # ffill adj closes & compute adj close logreturns
    # use code above to get indexes of wrong ones, raise warnings, set to 0
    #
    # check close vs adj close, there should be only dividends (with y finance)
    #
    # throw out opens that are not in [low, high]
    #
    # apply similar logic (perhaps using total lrets for the stddev) for
    # open-close , close-high , close-low, throw out open/low/close not OK
    #
    # fill
    #
    # compute open-open total returns, then check with same logic for errors
    #
    # when doing append, make past data adhere to same format: recompute adj close
    #
    # could use volumes as well, if there are jumps in price due to
    # splits not recorded, then price * volume should be more stable
    #
    #

    def _nan_impossible(self, data):
        """Set impossible values to NaN."""

        # print(data)
        # print(data.isnull().sum())

        # nan-out nonpositive prices
        for column in ["open", "close", "high", "low", "adjclose"]:
            bad_indexes = data.index[data[column] <= 0]
            if len(bad_indexes) > 0:
                logger.warning(
                    '%s("%s") has non-positive %s prices on timestamps: %s,'
                    + ' setting to nan',
                    self.__class__.__name__, self.symbol, column, bad_indexes)
                data.loc[bad_indexes, column] = np.nan

        # nan-out negative volumes
        bad_indexes = data.index[data["volume"] < 0]
        if len(bad_indexes) > 0:
            logger.warning(
                '%s("%s") has negative volumes on timestamps: %s,'
                + ' setting to nan',
                self.__class__.__name__, self.symbol, bad_indexes)
            data.loc[bad_indexes, "volume"] = np.nan

        # all infinity values are nans
        if np.isinf(data).sum().sum() > 0:
            logger.warning(
                '%s("%s") has +/- infinity values, setting those to nan',
                self.__class__.__name__, self.symbol)
            data.iloc[:, :] = np.nan_to_num(
                data.values, copy=True, nan=np.nan, posinf=np.nan,
                neginf=np.nan)

        # print(data)
        # print(data.isnull().sum())

        # TODO: these can be made smarter (sometimes the open is clearly wrong)

        # if low is not the lowest, set it to nan
        bad_indexes = data.index[
            data['low'] > data[['open', 'high', 'close']].min(1)]
        if len(bad_indexes) > 0:
            logger.warning(
                '%s("%s") low prices are not the lowest on timestamps: %s,'
                + ' setting to nan',
                self.__class__.__name__, self.symbol, bad_indexes)
            data.loc[bad_indexes, "low"] = np.nan

        # if high is not the highest, set it to nan
        bad_indexes = data.index[
            data['high'] < data[['open', 'high', 'close']].max(1)]
        if len(bad_indexes) > 0:
            logger.warning(
                '%s("%s") high prices are not the highest on timestamps: %s,'
                + ' setting to nan',
                self.__class__.__name__, self.symbol, bad_indexes)
            data.loc[bad_indexes, "high"] = np.nan

        # print(data)
        # print(data.isnull().sum())

    def _fill_easy(self, data):
        """Make easy fills."""

        # print(data)
        # print(data.isnull().sum())

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

    def _compute_total_returns(self, data):
        """Compute total open-to-open returns."""

        # print(data)
        # print(data.isnull().sum())

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

        # print(data)
        # print(data.isnull().sum())

    def _clean(self, data):
        """Clean Yahoo Finance open-close-high-low-volume-adjclose data."""

        self._nan_impossible(data)

        self._fill_easy(data)

        self._compute_total_returns(data)

        # eliminate adjclose column
        del data["adjclose"]

        # eliminate last period's intraday data
        data.loc[data.index[-1],
            ["high", "low", "close", "return", "volume"]] = np.nan

        return data

    def _quality_check(self, data):
        """Analyze quality of the OHLCV-TR data."""

        # zero volume
        zerovol_idx = data.index[data.volume == 0]
        if len(zerovol_idx) > 0:
            logger.warning(
                '%s("%s") has volume equal to zero for timestamps: %s',
                self.__class__.__name__, self.symbol, zerovol_idx)

        def print_extreme(logreturns, name, sigmas=50):

            # TODO: choose
            m, s = logreturns.median(), np.sqrt((logreturns**2).median())
            normalized = (logreturns - m)/s

            # normalized = logreturns / logreturns.rolling(252).std().shift(1)

            extremereturn_idx = normalized.index[np.abs(normalized) > sigmas]
            if len(extremereturn_idx) > 0:
                logger.warning(
                    '%s("%s") has extreme %s (~%s sigmas) for timestamps: %s',
                    self.__class__.__name__, self.symbol, name, sigmas,
                    extremereturn_idx)

        # extreme logreturns
        logreturns = np.log(1 + data['return']).dropna()
        print_extreme(logreturns, 'total returns')

        # extreme open2close
        open2close = np.log(data['close']) - np.log(data['open']).dropna()
        print_extreme(open2close, 'open to close returns')

        # extreme open2high
        open2high = np.log(data['high']) - np.log(data['open']).dropna()
        print_extreme(open2high, 'open to high returns')

        # extreme open2low
        open2low = np.log(data['low']) - np.log(data['open']).dropna()
        print_extreme(open2low, 'open to low returns')

class YahooFinance(OHLCVTR):
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
        except KeyError as exc:
            raise DataError(f'Yahoo finance download of {ticker} failed.'
                + ' Json:', str(res.json())) from exc # pragma: no cover

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

        self._quality_check(data)
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
            return pd.to_numeric(pd.read_csv(
                self.URL + f'?id={symbol}',
                index_col=0, parse_dates=[0])[symbol], errors='coerce')
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
