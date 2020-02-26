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
