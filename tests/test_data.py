"""
Copyright 2023- The Cvxportfolio Contributors

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

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio.data import YfinanceBase, LocalDataStore, Yfinance, FredBase


def test_yfinance_download():
    """Test YfinanceBase."""

    data = YfinanceBase().download('AAPL', start = '2023-04-01', end = '2023-04-15')
    print(data)
    print(data.loc['2023-04-10']['Return'])
    print(data.loc['2023-04-11', 'Open'] / data.loc['2023-04-10', 'Open'] - 1)
    assert np.isclose(data.loc['2023-04-10', 'Return'], data.loc['2023-04-11', 'Open'] / data.loc['2023-04-10', 'Open'] - 1)
    assert np.isnan(data.iloc[-1]['Close'])
    

def test_local_store_series(tmp_path):
    """Test storing and retrieving of a Series with datetime index."""
    store = LocalDataStore(tmp_path)
    for data in [pd.Series(0., pd.date_range('2020-01-01', '2020-01-10'), name='prova1'),
                 pd.Series(3, pd.date_range('2020-01-01', '2020-01-10'), name='prova2'),
                 pd.Series('ciao', pd.date_range('2020-01-01', '2020-01-02', freq='H'), name='prova3')]:
        print(data)
        print(data.index.dtype)
        print(data.dtypes)

        store.store(data.name, data)
        data1 = store.load(data.name)
        print(data1)
        print(data1.index.dtype)
        print(data1.dtypes)

        assert data.name == data1.name
        assert all(data == data1)
        assert all(data.index == data1.index)
        assert data.dtypes == data1.dtypes
        
    assert store.load('blahblah') is None
        
        
def test_local_store_dataframe(tmp_path):
    """Test storing and retrieving of a DataFrame with datetime index."""
    store = LocalDataStore(tmp_path)
    index = pd.date_range('2020-01-01', '2020-01-02', freq='H')
    data = {'one':range(len(index)), 'two':np.arange(len(index))/19., 'three':['hello']*len(index), 'four':[np.nan]*len(index)}
    data = pd.DataFrame(data, index=index)
    print(data)
    print(data.index.dtype)
    print(data.dtypes)

    store.store('example', data)
    data1 = store.load('example')
    print(data1)
    print(data1.index.dtype)
    print(data1.dtypes)

    assert all(data == data1)
    assert all(data.index == data1.index)
    assert all(data.dtypes == data1.dtypes)    
    

def test_yfinance(tmp_path):
    """Test yfinance ability to store and retrieve."""
    
    store = Yfinance(tmp_path)
    data = store.update_and_load('ZM')
    
    print(data)
    
    assert np.isclose(data.loc['2023-04-05', 'Return'], data.loc['2023-04-06', 'Open'] / data.loc['2023-04-05', 'Open'] - 1)
    
    data1 = store.update_and_load('ZM')
    print(data1)
    
    assert np.isnan(data1.iloc[-1]['Close'])
    
    print((data1.iloc[:len(data)-1].Return- data.iloc[:-1].Return).describe().T)
    
    assert np.allclose(data1.iloc[:len(data)-1].Return, data.iloc[:-1].Return)    
    
    
def test_fred_base():
    data = FredBase().download("DFF")
    assert data['2023-04-06'] == 4.83
    assert data.index[0] == pd.Timestamp('1954-07-01')
    print(data)
    olddata = data.iloc[:-123]
    newdata = FredBase().download("DFF", olddata)
    assert np.all(data == newdata)
    
