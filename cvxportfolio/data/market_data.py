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
"""This module defines :class:`MarketData` and derived classes."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataError
from ..utils import (hash_, make_numeric, periods_per_year_from_datetime_index,
                     resample_returns, set_pd_read_only)
from .symbol_data import *
from .symbol_data import OLHCV

logger = logging.getLogger(__name__)

__all__ = ['DownloadedMarketData', 'MarketData', 'UserProvidedMarketData']


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
        :rtype: (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series)
        """
        raise NotImplementedError # pragma: no cover

    def universe_at_time(self, t):
        """Return trading universe at given time.

        Base implementation simply returns the index of ``current_returns``
        returned by :meth:`serve`. Typically a more efficient implementation
        can be made available.

        :param t: Trading time. It must be included in the timestamps returned
            by :meth:`trading_calendar`.
        :type t: pandas.Timestamp

        :returns: Trading universe at time ``t``.
        :rtype: pd.Index
        """
        return self.serve(t)[1].index # pragma: no cover

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
        :rtype: pd.DatetimeIndex
        """
        raise NotImplementedError # pragma: no cover

    @property
    def periods_per_year(self):
        """Average trading periods per year.

        :rtype: int
        """
        raise NotImplementedError # pragma: no cover

    @property
    def full_universe(self):
        """Full universe, which might not be available for trading.

        :returns: Full universe.
        :rtype: pd.Index
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
        online_usage = False, universe_selection_in_time=None):
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
        self._validate_universe_selection(universe_selection_in_time)
        self.universe_selection_in_time = universe_selection_in_time

    def _validate_universe_selection(self, universe_selection_in_time):
        """Validate input of universe_selection_in_time."""
        if universe_selection_in_time is None:
            return
        if isinstance(universe_selection_in_time, pd.DataFrame
            ) and universe_selection_in_time.columns.equals(
                self.full_universe[:-1]) and isinstance(
                    universe_selection_in_time.index, pd.DatetimeIndex) and (
                        np.all(universe_selection_in_time.dtypes == bool)):
            return
        raise DataError("universe_selection_in_time must be a DataFrame with"
            " the full universe (minus cash) as columns and datetime index."
            " Dtypes should be boolean. NaNs are not allowed.")

    def _mask_dataframes(self, mask):
        """Mask internal dataframes if necessary."""
        if (self._mask is None) or not np.all(self._mask == mask):
            logger.info("Masking internal %s dataframes.",
                self.__class__.__name__)
            colmask = self.returns.columns[mask]
            # self._masked_returns = set_pd_read_only(
            #     pd.DataFrame(self.returns.iloc[:, mask], copy=True))
            self._masked_returns = set_pd_read_only(
               pd.DataFrame(self.returns.loc[:, colmask], copy=True))
            # self._masked_returns = set_pd_read_only(
            #     pd.DataFrame(np.array(self.returns.values[:, mask]),
            #         index=self.returns.index, columns=colmask))
            if not self.volumes is None:
                # self._masked_volumes = set_pd_read_only(
                #     pd.DataFrame(self.volumes.iloc[:, mask[:-1]], copy=True))
                self._masked_volumes = set_pd_read_only(
                    pd.DataFrame(self.volumes.loc[:, colmask[:-1]], copy=True))
                # self._masked_volumes = set_pd_read_only(
                #     pd.DataFrame(np.array(self.volumes.values[:, mask[:-1]]),
                #         index=self.volumes.index, columns=colmask[:-1]))
            if not self.prices is None:
                # self._masked_prices = set_pd_read_only(
                #     pd.DataFrame(self.prices.iloc[:, mask[:-1]], copy=True))
                self._masked_prices = set_pd_read_only(
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
        past_returns = set_pd_read_only(
            pd.DataFrame(self._masked_returns.iloc[:tidx]))
        current_returns = set_pd_read_only(
            pd.Series(self._masked_returns.iloc[tidx]))

        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            past_volumes = set_pd_read_only(
                pd.DataFrame(self._masked_volumes.iloc[:tidx]))
            current_volumes = set_pd_read_only(
                pd.Series(self._masked_volumes.iloc[tidx]))
        else:
            past_volumes = None
            current_volumes = None

        if not self.prices is None:
            tidx = self.prices.index.get_loc(t)
            current_prices = set_pd_read_only(
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

        logger.info('Adding cash column to the historical returns dataframe.')

        if cash_key == 'cash':
            self.returns['cash'] = 0.
            return

        if not cash_key in RATES:
            raise NotImplementedError(
                'Currently the only data pipelines built are for cash_key'
                f' in {list(RATES)}, or "cash".')

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
        tmp = pd.concat(
            [self.returns, cash_returns_per_period], sort=True, axis=1)
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

        valid_universe_mask = pd.Series(True, self.full_universe)

        # apply min_history filtering
        past_returns = self.returns.loc[self.returns.index < t]
        valid_universe_mask.iloc[
            :-1] &= past_returns.iloc[:, :-1].count() >= self._min_num_obs

        # apply universe selection
        if self.universe_selection_in_time is not None:
            valid_universe_mask.iloc[
                :-1] &= self.universe_selection_in_time.loc[
                    self.universe_selection_in_time.index <= t].iloc[-1]

        # if not running online remove current NaNs
        if not self.online_usage:
            valid_universe_mask &= ~self.returns.loc[t].isnull()
            if np.isnan(self.returns.loc[t].iloc[-1]):
                raise DataError(f'The cash return is nan at time {t}.')

        if sum(valid_universe_mask) <= 1:
            raise DataError(
                f'The trading universe at time {t} has size less or equal'
                + ' than one, i.e., only the cash account. There are probably '
                + ' issues with missing data in the provided market returns.')
        return valid_universe_mask

    def _set_read_only(self):
        """Set internal dataframes to read-only."""

        self.returns = set_pd_read_only(self.returns)

        if not self.prices is None:
            self.prices = set_pd_read_only(self.prices)

        if not self.volumes is None:
            self.volumes = set_pd_read_only(self.volumes)

    @property
    def _earliest_backtest_start(self):
        """Earliest date at which we can start a backtest."""
        return self.returns.iloc[:, :-1].dropna(how='all').index[
            self._min_num_obs]

    sampling_intervals = {
        'weekly': 'W-MON', 'monthly': 'MS', 'quarterly': 'QS', 'annual': 'YS'}

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

        # we nan-out the first non-nan element of every col
        for col in self.returns.columns[:-1]:
            self.returns.loc[
                    (~(self.returns[col].isnull())).idxmax(), col] = np.nan

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

            # we nan-out the first non-nan element of every col
            for col in self.volumes.columns:
                self.volumes.loc[
                    (~(self.volumes[col].isnull())).idxmax(), col] = np.nan

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

            # we nan-out the first non-nan element of every col
            for col in self.prices.columns:
                self.prices.loc[
                    (~(self.prices[col].isnull())).idxmax(), col] = np.nan

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
    def _min_num_obs(self):
        """Min history expressed in periods.

        :returns: How many non-null elements of the past returns for a given
            name are required to include it.
        :rtype: int
        """
        return int(np.round(self.periods_per_year * (
            self._min_history_timedelta / pd.Timedelta('365.24d'))))


class UserProvidedMarketData(MarketDataInMemory):
    """User-provided market data.

    .. versionadded:: 1.3.0

        The new parameter ``universe_selection_in_time`` used to optionally
        exclude assets from the trading universe at different points in time.

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
        of the provided returns, it will be added. In that case you should
        make sure your provided dataframes have a timezone aware datetime
        index. Its returns are the risk-free rate.
        If not in the original dataframe, choice of ``USDOLLAR``, ``EURO``,
        ``JPYEN``, ``GBPOUND`` (Cvxportfolio downloads the historical central
        bank rates from FRED), or ``cash``, which sets the cash returns to 0.
        Default ``USDOLLAR``.
    :type cash_key: str
    :param online_usage: Disable removal of assets that have ``np.nan`` returns
        for the given time. Default False.
    :type online_usage: bool
    :param universe_selection_in_time: Boolean dataframe used to specify
        which assets are to be included in the trading universe at each point
        in time. The columns are the full universe (same columns as ``prices``,
        ``volumes``, or ``returns`` without the last column, cash). The index
        is datetime and, differently from the usual convention, needs not to
        to be the same as the index of the other dataframes: at each point
        in time of a back-test the *last* valid observation (before, or at, the
        trading time) is selected. (You still need to provide a timezoned index
        if the returns' index is, otherwise time comparisons can't be done.)
        The entries are boolean; ``True`` means that the corresponding asset
        can be invested in at the time, ``False`` that it can't. Note that this
        is more fundamental than imposing time-varying position limit, for
        example, in an optimization-based policy. Non-investable assets are
        removed by the :class:`MarketSimulator`; their positions converted to
        cash, the :class:`result.BacktestResult` dataframes will have
        ``np.nan`` on those dates for those assets', and the policy is
        re-compiled without those assets. You shouldn't use it to make frequent
        changes to the trading universe (if that's your usecase), but rather
        impose time-varying position limits via :class:`constraints.MinWeights`
        and :class:`constraints.MaxWeights`. Also, note that
        the filtering implied by ``min_history`` is still applied (in addition
        to this), as well as non-``nan`` returns for the period (which can be
        disabled by ``online_usage``). Default, None, don't use this filtering.
    :type universe_selection_in_time: pd.DataFrame or None
    """

    # pylint: disable=too-many-arguments
    def __init__(self, returns, volumes=None, prices=None,
                 copy_dataframes=True, trading_frequency=None,
                 min_history=pd.Timedelta('365.24d'),
                 base_location=BASE_LOCATION,
                 grace_period=pd.Timedelta('1d'),
                 cash_key='USDOLLAR',
                 online_usage=False,
                 universe_selection_in_time=None):

        if returns is None:
            raise SyntaxError(
                "If you don't specify a universe you should pass `returns`.")

        self.base_location = Path(base_location)
        self.cash_key = cash_key

        self.returns = pd.DataFrame(
            make_numeric(returns), copy=copy_dataframes)
        self._validate_user_provided_returns()
        self.volumes = volumes if volumes is None else\
            pd.DataFrame(make_numeric(volumes), copy=copy_dataframes)
        self.prices = prices if prices is None else\
            pd.DataFrame(make_numeric(prices), copy=copy_dataframes)

        if cash_key != returns.columns[-1]:
            self._add_cash_column(cash_key, grace_period=grace_period)
        assert self.returns.columns[-1] == cash_key

        # this is mandatory
        super().__init__(
            trading_frequency=trading_frequency,
            base_location=base_location,
            cash_key=cash_key,
            min_history=min_history,
            online_usage=online_usage,
            universe_selection_in_time=universe_selection_in_time)

    def _validate_user_provided_returns(self):
        """Log warning if returns have incorrect nan structure.

        Each column of returns should only have nans at the top and/or at the
        bottom. Things work anyway, but it's probably an error by the user.
        """
        changes_nan_status = self.returns.isnull().diff().sum()
        changes_nan_status += ~self.returns.iloc[0].isnull()
        bad_assets = changes_nan_status[changes_nan_status > 2]
        if len(bad_assets) > 0:
            logger.warning(
                'In the user-provided returns, assets %s seem to have'
                ' incorrect missing values structure. For each asset, missing'
                ' returns should only be at the start and/or at the end.'
                ' You should use universe_selection_in_time if you wanted to'
                ' specify changes of universe in time.', bad_assets)


class DownloadedMarketData(MarketDataInMemory):
    """Market data that is downloaded.

    .. versionadded:: 1.3.0

        The new parameter ``universe_selection_in_time`` used to optionally
        exclude assets from the trading universe at different points in time.

    :param universe: List of names as understood by the data source
        used, *e.g.*, ``['AAPL', 'GOOG']`` if using the default
        Yahoo Finance data source.
    :type universe: list
    :param datasource: The data source used.
    :type datasource: str or :class:`SymbolData` class
    :param cash_key: Name of the cash account, its rates will be
        added as last column of the returns, as the
        risk-free rate. Choice of ``USDOLLAR``, ``EURO``,``JPYEN``, ``GBPOUND``
        (Cvxportfolio downloads the historical central bank rates from FRED),
        or ``cash``, which sets the cash returns to 0. Default ``USDOLLAR``.
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
    :param universe_selection_in_time: Boolean dataframe used to specify
        which assets are to be included in the trading universe at each point
        in time. The columns are the full ``universe``. The index
        is datetime and, differently from the usual convention, needs not to
        to be the same as the index of the other dataframes: at each point
        in time of a back-test the *last* valid observation (before, or at, the
        trading time) is selected. (You still need to provide a timezoned index
        if the returns' index is, otherwise time comparisons can't be done.)
        The entries are boolean; ``True`` means that the corresponding asset
        can be invested in at the time, ``False`` that it can't. Note that this
        is more fundamental than imposing time-varying position limit, for
        example, in an optimization-based policy. Non-investable assets are
        removed by the :class:`MarketSimulator`; their positions converted to
        cash, the :class:`result.BacktestResult` dataframes will have
        ``np.nan`` on those dates for those assets', and the policy is
        re-compiled without those assets. You shouldn't use it to make frequent
        changes to the trading universe (if that's your usecase), but rather
        impose time-varying position limits via :class:`constraints.MinWeights`
        and :class:`constraints.MaxWeights`. Also, note that
        the filtering implied by ``min_history`` is still applied (in addition
        to this), as well as non-``nan`` returns for the period (which can be
        disabled by ``online_usage``). Default, None, don't use this filtering.
    :type universe_selection_in_time: pd.DataFrame or None
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
                 online_usage=False,
                 universe_selection_in_time=None):
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
        self.online_usage = online_usage
        self._remove_missing_recent()

        # this is mandatory
        super().__init__(
            trading_frequency=trading_frequency,
            base_location=base_location,
            cash_key=cash_key,
            min_history=min_history,
            online_usage=online_usage,
            universe_selection_in_time=universe_selection_in_time)

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

        if issubclass(self.datasource, OLHCV):
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
            # pylint: disable=logging-not-lazy
            logger.warning(
                ('Missing values detected in recent lines.'
                    if self.online_usage else
                'Removing some recent lines because there are missing values.')
                + ' This is most probably an error with the data source!'
                + ' The issue is with symbol(s) %s',
                self.prices.columns[self.prices.iloc[-5:].isnull().any()])
            if not self.online_usage:
                drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
                logger.warning('Dropping at index %s', drop_at)
                self.returns = self.returns.loc[self.returns.index <= drop_at]
                if self.prices is not None:
                    self.prices = self.prices.loc[self.prices.index <= drop_at]
                if self.volumes is not None:
                    self.volumes = self.volumes.loc[
                        self.volumes.index <= drop_at]

        # for consistency we must also nan-out the last row
        # of returns and volumes
        self.returns.iloc[-1] = np.nan
        if self.volumes is not None:
            self.volumes.iloc[-1] = np.nan

        # prices could have issues too, they are only used for rounding
        # so this should be OK
        if self.prices is not None:
            self.prices.iloc[-1] = self.prices.iloc[-1].fillna(
                self.prices.iloc[-2])

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
