"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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
from pandas_datareader import data
import pandas as pd

RISKFREENAME = "FEDFUNDS"
RISKFREESOURCE = 'fred'
STOCKSOURCE = 'yahoo'
CLOSEFIELD = 'Adj Close'


def get_stock_daily_data(tickers, start_day, end_day):
    stock_data = data.DataReader(tickers, STOCKSOURCE, start_day, end_day)
    log_returns = np.log(stock_data[CLOSEFIELD]).diff()
    returns = np.exp(log_returns)[1:] - 1
    sigma = (stock_data.High - stock_data.Low) / stock_data.High
    volume = (stock_data.Volume * stock_data[CLOSEFIELD])
    return returns, sigma, volume


def get_risk_free(time_index, PPY=250):
    """
    Args:
        time_index: a Pandas time index object
        PPY: number of periods in a year. defaults to 250 (for daily)

    Returns:
        Pandas Series indexed by time_index with risk free per period return
    """
    risk_free = data.DataReader(RISKFREENAME, RISKFREESOURCE,
                                time_index[0] - pd.Timedelta('31 days'),  # data is available only once a month
                                time_index[-1] + pd.Timedelta('31 days'))
    risk_free /= 100.  # expressed in percent
    risk_free = risk_free.reindex(time_index, method='ffill')
    risk_free /= PPY  # because it's annualized
    return risk_free


def get_capitalization_weights(tickers):
    """

    Args:
        tickers: list of stocks (making up the benchmark)

    Returns:
        pandas Series of weights (sum to one) of market cap

    """
    from pandas_datareader.data import get_quote_yahoo
    from pandas_datareader.yahoo.quotes import _yahoo_codes

    _yahoo_codes.update({'MarketCap': 'j1'})
    market_cap = get_quote_yahoo(tickers)['MarketCap']

    def str_to_val(mcap_str):
        end_idx = mcap_str.index('B')
        val = float(str(mcap_str[:end_idx]))
        return val

    benchmark = market_cap.apply(str_to_val)
    benchmark /= benchmark.sum()
    return benchmark


if __name__ == "__main__":
    import datetime

    start_day = datetime.datetime(2014, 1, 10)
    end_day = datetime.datetime(2014, 1, 15)

    tickers = ['AAPL', 'GOOG']

    returns, sigma, volume = get_stock_daily_data(tickers, start_day, end_day)

    risk_free = get_risk_free(returns.index)

    print('returns', returns)
    print('sigma', sigma)
    print('volume', volume)
    print('risk free', risk_free)

    print('weights cap', get_capitalization_weights(tickers))
