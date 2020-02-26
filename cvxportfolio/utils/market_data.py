"""
Copyright 2020 Enzo Busseti.

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

from datetime import datetime
import logging

import numpy as np
import pandas as pd
import pandas_datareader as pdr

__all__ = ['get_adjusted_symbol', 'get_featurized_symbol']


def get_adjusted_symbol(yahoo_symbol, start, end):
    """Utility function to get adjusted data from yahoo finance."""

    data = pdr.get_data_yahoo(symbols=yahoo_symbol,
                              start=start, end=end)

    adjusted_data = pd.DataFrame(index=data.index)
    adjusted_data['Volume'] = data.Volume
    adjusted_data['Close'] = data['Adj Close']
    correction = (adjusted_data['Close'] / data['Close'])
    adjusted_data['Open'] = correction * data['Open']
    adjusted_data['Min'] = correction * data['Low']
    adjusted_data['Max'] = correction * data['High']

    return adjusted_data


def _featurize_barchart(data, Open='Open', Close='Close',
                        AdjClose='Adj Close',
                        Max='High', Min='Low',
                        Volume='Volume',
                        extra_features=False):
    """CVXPortfolio conventional featurization of standard (e.g., daily) barchart data.

    Specify the correct names of the columns in the `data` pd.DataFrame.

    Notes:
        The provided volume is assumed to be expressed in number of units traded.
        The returned price is *not adjusted* for stock splits or dividends,
        it is used for rounding trades in the simulator.
    """

    logopen = np.log(data[Open])
    logclose = np.log(data[Close])
    logmax = np.log(data[Max])
    logmin = np.log(data[Min])

    featurized = pd.DataFrame(index=data.index)

    featurized['price'] = data[Open]  # for rounding trades
    featurized['log_volume'] = np.log(
        data[Open] * data[Volume] + 1)  # for TCost model
    ex_market_ret = (np.log(data[AdjClose]) - logclose).diff().shift(-1)
    featurized['period_return'] = logclose - logopen
    featurized[
        'ex_period_return'] = logopen.shift(-1) - logclose + ex_market_ret
    featurized['return'] = featurized['period_return'] + \
        featurized['ex_period_return']  # for simulator
    featurized['range'] = (logmax - logmin)  # for TCost model

    if extra_features:
        featurized['max_diff'] = logmax - logopen
        featurized['min_diff'] = logmin - logopen
        featurized['sigma2'] = featurized['return']**2

    else:
        del featurized['period_return']
        del featurized['ex_period_return']

    # mini test
    total_return_2 = logopen.diff().shift(-1) + ex_market_ret
    assert np.allclose(featurized['return'].iloc[
                       :-1], total_return_2.iloc[:-1])

    return featurized


def get_featurized_symbol(yahoo_symbol, start, end,
                          extra_features=False):

    data = pdr.get_data_yahoo(symbols=yahoo_symbol,
                              start=start, end=end)

    return _featurize_barchart(
        data,
        extra_features=extra_features)


DATA_OVERLAP = pd.Timedelta('2d')
CASH = 'USDOLLAR'
logger = logging.getLogger(__name__)


class MarketData(object):
    """A picklable object that downloads and serves market data."""

    def __init__(self, symbols, history='20d', extra_features=False):
        if not len(symbols):
            raise Exception('`symbols` must be an iterable of valid symbols.')
        self._symbols = sorted(symbols)
        self._extra_features = extra_features
        self._data = {}
        self._history = pd.Timedelta(history)
        self._initial_download(self._symbols,
                               start=datetime.today() - self._history,
                               end=datetime.today())
        self._update_index()
        self._download_cash()

    def _update_index(self):
        self._index = pd.DatetimeIndex(sorted(set(
            sum([list(self._data[symbol].index)
                 for symbol in self._symbols], []))))
        # print(self._index)

    @property
    def symbols(self):
        return self._symbols

    def _initial_download(self, symbols, start=None, end=None):
        if start is None:
            start = self._index[0]
        if end is None:
            end = self._index[-1]
        for symbol in symbols:
            self._data[symbol] = self._get_featurized_symbol(
                symbol, start, end)

    def _download_cash(self):
        self._data[CASH] = self._get_us_effective_rate(
            self._index[0] - pd.Timedelta('2d'), self._index[-1]).loc[
            self._index]

    def add_symbols(self, symbols):
        self._symbols = sorted(list(set(self._symbols).union(symbols)))
        self._initial_download(symbols)

    def remove_symbols(self, symbols):
        self._symbols = sorted(list(set(self._symbols).difference(symbols)))
        for symbol in symbols:
            del self._data[symbol]

    def update_store(self):

        download_start = self._index[-1] - DATA_OVERLAP
        download_end = datetime.today()

        for symbol in self._symbols:
            downloaded_chunk = self._get_featurized_symbol(
                symbol, download_start, download_end)
            self._data[symbol].loc[downloaded_chunk.index] = downloaded_chunk

        self._update_index()

    def _get_us_effective_rate(self, start, end):
        logger.info(f'Downloading US federal fund effective rate from {start} to {end}')
        return pdr.get_data_fred('DTB3',
                                 start=start,
                                 end=end)['DTB3'].fillna(method='ffill')

    def _get_featurized_symbol(self, symbol, start, end):
        logger.info(f'Downloading {symbol} from {start} to {end}')
        return get_featurized_symbol(symbol, start=start,
                                     end=end,
                                     extra_features=self._extra_features)

    @property
    def all_market_data(self):
        return self.market_data_up_to_time(datetime.today())

    def market_data_up_to_time(self, t):
        data = pd.concat([self._data[symbol].loc[
            self._data[symbol].index <= t].rename(
            columns={col: f'{symbol}_{col}' for col in
                     self._data[symbol].columns})
            for symbol in self._symbols],
            # + ([pd.DataFrame({'USDOLLAR': self._data['USDOLLAR']})]
            #     if self._cash   == 'USDOLLAR' else []),
            axis=1)
        return data

    @property
    def volumes(self):
        return np.exp(pd.DataFrame({symbol: self._data[symbol]['log_volume']
                                    for symbol in self._symbols}))

    @property
    def prices(self):
        return pd.DataFrame({symbol: self._data[symbol]['price']
                             for symbol in self._symbols})

    @property
    def returns(self):
        rets = pd.DataFrame({symbol: self._data[symbol]['return']
                             for symbol in self._symbols})
        rets[CASH] = self._data[CASH] / 100.
        rets[CASH] = rets[CASH].fillna(method='ffill').fillna(method='bfill')
        return rets

    @property
    def ranges(self):
        return pd.DataFrame({symbol: self._data[symbol]['range']
                             for symbol in self._symbols})

if __name__ == '__main__':

    import pickle
    import sys

    mar = MarketData(symbols=['IVV', 'IAU'])

    print(mar.returns)
    print(mar.volumes)
    print(mar.ranges)
    print(mar.prices)
    print(mar.all_market_data)

    print(mar.symbols)

    mar.add_symbols(['AAPL'])

    print(mar.returns)

    mar.remove_symbols(['IVV'])
    print(mar._data.keys())

    print(mar.symbols)

    print(mar.returns)
    mar_string = pickle.dumps(mar)
    print(sys.getsizeof(mar_string))
    new_mar = pickle.loads(mar_string)
    print(new_mar.returns)

    print(new_mar.market_data_up_to_time('2020-02-14'))

    new_mar.update_store()
    print(new_mar.returns)

    # print(new_mar._data)
