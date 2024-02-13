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
"""This module defines :class:`SymbolData` and derived classes."""

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
from ..utils import set_pd_read_only

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
        return set_pd_read_only(self._preload(self._load_raw()))

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

        Checks (which raise warnings):

        #. Elements of data are NaN (skipping last row)
        #. Update is not append-only. For dataframes check all elements other
        than last row of the data which was there before, and for that last
        row, only the open price. For Series that doesn't matter, check that
        last element is the same.

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
                            current.iloc[:-1], equal_nan=True)):
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
        # this should have become superflous
        except KeyError: # pragma: no cover
            logger.error("%s update of %s could not be checked for"
                + " append-only edits. Was there a DST change?",
                self.__class__.__name__, self.symbol) # pragma: no cover
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
        raise NotImplementedError # pragma: no cover

    def _preload(self, data):
        """Prepare data to serve to the user.

        This method can be redefined by derived classes.

        :param data: The data returned by the storage backend.
        :type data: pandas.Series or pandas.DataFrame
        :rtype: pandas.Series or pandas.DataFrame
        """
        return data # pragma: no cover


#
# Yahoo Finance.
#

def _timestamp_convert(unix_seconds_ts):
    """Convert a UNIX timestamp in seconds to a pandas.Timestamp."""
    return pd.Timestamp(unix_seconds_ts*1E9, tz='UTC')

# Anomalous, extreme, dubious logreturns filtering.

def _median_scale_around(lrets, window):
    """Median absolute logreturn in a window around each timestamp."""
    return np.abs(lrets).rolling(window, center=True, min_periods=1).median()

def _mean_scale_around(lrets, window):
    """Root mean squared logreturn in a window around each timestamp.

    We need a few operations because we skip the observation itself
    """
    sum = (lrets**2).rolling(window, center=True, min_periods=2).sum()
    count = lrets.rolling(window, center=True, min_periods=2).count()
    return np.sqrt((sum - lrets**2) / (count - 1))

def _unlikeliness_score(
        test_logreturns, reference_logreturns, scaler, windows):
    """Find problematic indexes for test logreturns compared w/ reference."""
    scaled = [
        np.abs(test_logreturns) / scaler(reference_logreturns, window)
        for window in windows]
    scaled = pd.DataFrame(scaled).T
    return scaled.min(axis=1)


class OLHCV(SymbolData): # pylint: disable=abstract-method
    """Base class for Open-Low-High-Close-Volume symbol data.

    This operates on a dataframe with columns

    .. code-block::

        ['open', 'low', 'high', 'close', 'volume']

    or

    .. code-block::

        ['open', 'low', 'high', 'close', 'volume', 'return']

    in which case the ``'return'`` column is not processed. It only matters in
    the :meth:`_preload`, method: if open-to-open returns are not present,
    we compute them there. Otherwise these may be total returns (which include
    dividends, ...) and they're dealt with in derived classes.
    """

    FILTERING_WINDOWS = (10, 20, 50, 100, 200)

    # remove open prices when open to close abs logreturn is larger than
    # this time the median absolute ones in FILTERING_WINDOWS around it
    THRESHOLD_OPEN_TO_CLOSE = 15

    # remove low/high prices when low/high to close abs logreturn larger than
    # this time the median absolute ones in FILTERING_WINDOWS centered on it
    THRESHOLD_LOWHIGH_TO_CLOSE = 20

    # log warning on _preload for abs logreturns (of 4 types) larger than this
    # time the root mean square in FILTERING_WINDOWS centered on it, without
    # the given observation itself
    THRESHOLD_WARN_EXTREME_LOGRETS = 10

    def _process(self, new_data, saved_data=None):
        """Base method for processing (cleaning) data.

        It operates on the ``new_data`` dataframe, which is the newly
        downloaded data. The ``saved_data`` dataframe is provided as well
        (None if there is none). It has the same columns, older timestamps
        (possibly overlapping with new_data at the end), and is **read only**:
        it is used as reference to help with the cleaning, it has already
        been cleaned.
        """

        ## Preliminaries
        ## Eliminate non-positive prices, infinity values.

        # NaN nonpositive prices
        for column in ["open", "close", "high", "low"]:
            self._nan_nonpositive_prices(new_data, column)

        # all infinity values to NaN
        self._set_infty_to_nan(new_data)

        ## Close price.
        ## We believe them (for now). We forward fill them if unavailable.

        # forward-fill close
        self._fillna_and_message(
            new_data, 'close', 'last available', filler='ffill')

        ## Volumes.
        ## We set negative to NaN, and fill with zeros.

        # NaN negative volumes
        self._nan_negative_volumes(new_data)

        # fill with zeros
        self._fillna_and_message(
            new_data, 'volume', 'zeros', filler='fillna', filler_arg=0.)

        ## Open price.
        ## We remove if lower than low, higher than high, or open to close
        ## logreturn is anomalous. Then we fill with close from day before.

        # NaN open if lower than low
        self._nan_open_lower_low(new_data)

        # NaN open if higher than high
        self._nan_open_higher_high(new_data)

        # NaN anomalous open prices
        self._nan_anomalous_prices(
            new_data, 'open', threshold=self.THRESHOLD_OPEN_TO_CLOSE,
            saved_data=saved_data)

        # fill open with close from day before
        self._fillna_and_message(
            new_data, 'open', 'close from period before', filler='fillna',
            filler_arg=new_data['close'].shift(1))

        ## Low price.
        ## We remove if higher than close or anomalous low to close logreturn.
        ## We fill them with min of open and close.

        # NaN low if higher than close
        self._nan_low_higher_close(new_data)

        # NaN low if higher than open (cleaned)
        self._nan_low_higher_open(new_data)

        # NaN anomalous low prices
        self._nan_anomalous_prices(
            new_data, 'low', threshold=self.THRESHOLD_LOWHIGH_TO_CLOSE,
            saved_data=saved_data)

        # fill low with min of open and close
        self._fillna_and_message(
            new_data, 'low', 'min of open and close', filler='fillna',
            filler_arg=new_data[['open', 'close']].min(axis=1))

        ## High price.
        ## We remove if lower than close or anomalous low to close logreturn.
        ## We fill them with max of open and close.

        # NaN high if lower than close
        self._nan_high_lower_close(new_data)

        # NaN high if lower than open (cleaned)
        self._nan_high_lower_open(new_data)

        # NaN anomalous high prices
        self._nan_anomalous_prices(
            new_data, 'high', threshold=self.THRESHOLD_LOWHIGH_TO_CLOSE,
            saved_data=saved_data)

        # fill high with max of open and close
        self._fillna_and_message(
            new_data, 'high', 'max of open and close', filler='fillna',
            filler_arg=new_data[['open', 'close']].max(axis=1))

        ## Some asserts
        assert new_data.iloc[1:].isnull().sum().sum() == 0
        assert np.all(
            new_data['low'].fillna(0.) <= new_data[
                ['open', 'high', 'close']].min(1))
        assert np.all(
            new_data['high'].fillna(np.inf) >= new_data[
                ['open', 'low', 'close']].max(1))

        return new_data

    def _fillna_and_message(
        self, data, col_name, message, filler='fillna', filler_arg=None):
        """Fill NaNs in column with chosen method and arg."""
        bad_indexes = data.index[data[col_name].isnull()]
        if len(bad_indexes) > 0:
            logger.warning(
                '%s("%s").data["%s"] has NaNs on timestamps: %s,'
                + ' filling them with %s.', self.__class__.__name__,
                self.symbol, col_name, bad_indexes, message)
            data[col_name] = getattr(data[col_name], filler)(filler_arg)

    def _nan_anomalous_prices(
            self, new_data, price_name, threshold, saved_data=None):
        """Set to NaN given price name on its anomalous logrets to close."""
        new_lr_to_close =\
            np.log(new_data['close']) - np.log(new_data[price_name])

        # if there is saved data, we use it to compute the logrets
        # also on the past, but we only NaN (if necessary) elements of
        # new data, so the scores computed on the past are not used
        if saved_data is None:
            all_lr_to_close = new_lr_to_close
        else:
            old_lr_to_close =\
                np.log(saved_data['close']) - np.log(saved_data[price_name])
            all_lr_to_close = pd.concat(
                [old_lr_to_close.loc[
                    old_lr_to_close.index < new_lr_to_close.index[0]],
                new_lr_to_close])
            # drop old data which we don't need
            all_lr_to_close = all_lr_to_close.iloc[
                -len(new_data) - max(self.FILTERING_WINDOWS):]

        # with this we skip over exact zeros (which come from some upstream
        # cleaning) and would throw the median off
        all_lr_to_close.loc[all_lr_to_close == 0] = np.nan
        score = _unlikeliness_score(
                all_lr_to_close, all_lr_to_close, scaler=_median_scale_around,
                windows=self.FILTERING_WINDOWS)
        self._nan_values(
            new_data, condition = score.loc[new_data.index] > threshold,
            columns_to_nan=price_name, message=f'anomalous {price_name} price')

    def _nan_values(self, data, condition, columns_to_nan, message):
        """Set to NaN in-place for indexing condition and chosen columns."""

        bad_indexes = data.index[condition]
        if len(bad_indexes) > 0:
            logger.warning(
                '%s("%s") has %s on timestamps: %s,'
                + ' setting to nan',
                self.__class__.__name__, self.symbol, message, bad_indexes)
            data.loc[bad_indexes, columns_to_nan] = np.nan

    def _nan_nonpositive_prices(self, data, prices_name):
        """Set non-positive prices (chosen price name) to NaN, in-place."""
        self._nan_values(
            data=data, condition = data[prices_name] <= 0,
            columns_to_nan = prices_name,
            message = f'non-positive {prices_name} prices')

    def _nan_negative_volumes(self, data):
        """Set negative volumes to NaN, in-place."""
        self._nan_values(
            data=data, condition = data["volume"] < 0,
            columns_to_nan = "volume", message = 'negative volumes')

    def _nan_open_lower_low(self, data):
        """Set open price to NaN if lower than low, in-place."""
        self._nan_values(
            data=data, condition = data['open'] < data['low'],
            columns_to_nan = "open",
            message = 'open price lower than low price')

    def _nan_open_higher_high(self, data):
        """Set open price to NaN if higher than high, in-place."""
        self._nan_values(
            data=data, condition = data['open'] > data['high'],
            columns_to_nan = "open",
            message = 'open price higher than high price')

    # def _nan_incompatible_low_high(self, data):
    #     """Set low and high to NaN if low is higher, in-place."""
    #     self._nan_values(
    #         data=data, condition = data['low'] > data['high'],
    #         columns_to_nan = ["low", "high"],
    #         message = 'low price higher than high price')

    def _nan_high_lower_close(self, data):
        """Set high price to NaN if lower than close, in-place."""
        self._nan_values(
            data=data, condition = data['high'] < data['close'],
            columns_to_nan = "high",
            message = 'high price lower than close price')

    def _nan_high_lower_open(self, data):
        """Set high price to NaN if lower than open, in-place."""
        self._nan_values(
            data=data, condition = data['high'] < data['open'],
            columns_to_nan = "high",
            message = 'high price lower than open price')

    def _nan_low_higher_close(self, data):
        """Set low price to NaN if higher than close, in-place."""
        self._nan_values(
            data=data, condition = data['low'] > data['close'],
            columns_to_nan = "low",
            message = 'low price higher than close price')

    def _nan_low_higher_open(self, data):
        """Set low price to NaN if higher than open, in-place."""
        self._nan_values(
            data=data, condition = data['low'] > data['open'],
            columns_to_nan = "low",
            message = 'low price higher than open price')

    def _set_infty_to_nan(self, data):
        """Set all +/- infty elements of data to NaN, in-place."""

        if np.isinf(data).sum().sum() > 0:
            logger.warning(
                '%s("%s") has +/- infinity values, setting those to nan',
                self.__class__.__name__, self.symbol)
            data.iloc[:, :] = np.nan_to_num(
                data.values, copy=True, nan=np.nan, posinf=np.nan,
                neginf=np.nan)

    def _warn_on_extreme_logreturns(self, logreturns, threshold, what):
        """Log warning if logreturns are extreme."""
        # with this we skip over exact zeros (which we assume come from some
        # cleaning) and would bias the mean down
        logreturns.loc[logreturns == 0] = np.nan
        score = _unlikeliness_score(
                logreturns, logreturns, scaler=_mean_scale_around,
                windows=self.FILTERING_WINDOWS)
        dubious_indexes = logreturns.index[score > threshold]
        if len(dubious_indexes) > 0:
            logger.warning(
                '%s("%s") has dubious %s for timestamps: %s',
                self.__class__.__name__, self.symbol, what, dubious_indexes)

    def _quality_check(self, data):
        """Log issues with the quality of data given to the user."""

        # zero volume
        zerovol_idx = data.index[data.volume == 0]
        if len(zerovol_idx) > 0:
            logger.info(
                '%s("%s") has volume equal to zero for timestamps: %s',
                self.__class__.__name__, self.symbol, zerovol_idx)

        # warn on extreme logreturns
        self._warn_on_extreme_logreturns(
            np.log(1 + data['return']), self.THRESHOLD_WARN_EXTREME_LOGRETS,
            'total open-to-open returns')

        # extreme open2close
        self._warn_on_extreme_logreturns(
            np.log(data['close']) - np.log(data['open']),
            self.THRESHOLD_WARN_EXTREME_LOGRETS, 'open to close returns')

        # extreme open2high
        self._warn_on_extreme_logreturns(
            np.log(data['high']) - np.log(data['open']),
            self.THRESHOLD_WARN_EXTREME_LOGRETS, 'open to high returns')

        # extreme open2low
        self._warn_on_extreme_logreturns(
            np.log(data['low']) - np.log(data['open']),
            self.THRESHOLD_WARN_EXTREME_LOGRETS, 'open to low returns')

    def _preload(self, data):
        """Prepare data for use by Cvxportfolio.

        We drop the `volume` column expressed in number of shares and
        replace it with `valuevolume` which is an estimate of the (e.g.,
        US dollar) value of the volume exchanged on the day.
        """

        # this is not used currently, but if we implement an interface to a
        # pure OLHCV data source there is no need to store the open-to-open
        # returns, they can be computed here
        if not 'return' in data.columns:
           data['return'] = data[
                'open'].pct_change().shift(-1) # pragma: no cover

        self._quality_check(data)

        # NaN intraday data
        data.loc[data.index[-1],
            ["high", "low", "close", "return", "volume"]] = np.nan

        # compute volume in cash units
        data["valuevolume"] = data["volume"] * data["open"]
        del data["volume"]

        return data

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
# when doing append, make past data adhere to same format: recompute adj
# close
# could use volumes as well, if there are jumps in price due to
# splits not recorded, then price * volume should be more stable
#
#


class YahooFinance(OLHCV):
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

    def _process(self, new_data, saved_data=None):
        """Process Yahoo Finance specific data, call parent's.

        Here we deal with the adjclose column, call OLHCV._process method, and
        compute total open-to-open returns.
        """

        ## Treat adjclose. We believe them (unless impossible).

        # all infinity values to NaN (repeat, but for adjclose)
        self._set_infty_to_nan(new_data)

        # NaN non-positive adj close
        self._nan_nonpositive_prices(new_data, "adjclose")

        # forward-fill adj close
        self._fillna_and_message(
            new_data, 'adjclose', 'last available', filler='ffill')

        ## OLHCV._process treats all columns other than adjclose
        super()._process(new_data, saved_data=saved_data)

        ## Compute total open-to-open returns

        # intraday logreturn
        intraday_logreturn = np.log(
            new_data["close"]) - np.log(new_data["open"])

        # close to close total logreturn
        close_to_close_total_logreturn = np.log(
            new_data["adjclose"]).diff().shift(-1)

        # open to open total logreturn
        open_to_open_total_logreturn = \
            close_to_close_total_logreturn + intraday_logreturn \
            - intraday_logreturn.shift(-1)

        # open to open total return
        new_data['return'] = np.exp(open_to_open_total_logreturn) - 1

        # eliminate adjclose column
        del new_data["adjclose"]

        return new_data

    @staticmethod
    def _get_data_yahoo(ticker, start='1900-01-01', end='2100-01-01'):
        """Get 1-day OLHC-AC-V from Yahoo finance.

        This is roughly equivalent to

        .. code-block::

            import yfinance as yf
            yf.download(ticker)

        But it does no caching of any sort; only a single request call,
        error checking (which result in exceptions going all the way to the
        user, in the current design), json parsing, and a minimal effort to
        restore the last timestamp. All processing and cleaning is done
        elsewhere.

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
            raise DataError(
                f'Yahoo finance download of {ticker} failed. Json:',
                str(res.json())) # pragma: no cover

        data = res.json()['chart']['result'][0]

        try:
            index = pd.DatetimeIndex(
                [_timestamp_convert(el) for el in data['timestamp']])

            df_result = pd.DataFrame(
                data['indicators']['quote'][0], index=index)
            df_result['adjclose'] = data[
                'indicators']['adjclose'][0]['adjclose']
        except KeyError as exc: # pragma: no cover
            raise DataError(f'Yahoo finance download of {ticker} failed.'
                + ' Json:', str(res.json())) from exc # pragma: no cover

        # last timestamp could be not timed to market open
        this_periods_open_time = _timestamp_convert(
            data['meta']['currentTradingPeriod']['regular']['start'])

        # this should be enough, but be careful
        if df_result.index[-1] > this_periods_open_time:
            index = df_result.index.to_numpy()
            index[-1] = this_periods_open_time
            df_result.index = pd.DatetimeIndex(index)

        # # last timestamp is probably broken (not timed to market open)
        # # we set its time to same as the day before, but this is wrong
        # # on days of DST switch. It's fine though because that line will be
        # # overwritten next update
        # if df_result.index[-1].time() != df_result.index[-2].time():
        #     tm1 = df_result.index[-2].time()
        #     newlast = df_result.index[-1].replace(
        #         hour=tm1.hour, minute=tm1.minute, second=tm1.second)
        #     df_result.index = pd.DatetimeIndex(
        #         list(df_result.index[:-1]) + [newlast])

        # these are all the columns, we simply re-order them
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
        # TODO this could be put at a lower class hierarchy
        if overlap < 2:
            raise SyntaxError(
                f'{self.__class__.__name__} with overlap smaller than 2'
                + ' could have issues with DST.')
        if (current is None) or (len(current) < overlap):
            updated = self._get_data_yahoo(symbol, **kwargs)
            logger.info('Downloading from the start.')
            result = self._process(updated)
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
        new = self._process(new)
        return pd.concat([current.iloc[:-overlap], new])


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
