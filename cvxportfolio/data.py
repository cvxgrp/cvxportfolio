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

import yfinance as yf
import pandas as pd


class BaseData:
    
    def load_raw(self, symbol):
        raise NotImplementedError
        
    def load(self, symbol):
        return self.preload(self.load_raw(symbol))
        
    def store(self, symbol, data):
        raise NotImplementedError
        
    def download(self, symbol, current=None):
        raise NotImplementedError
        
    def update_and_load(self, symbol):
        current = self.load_raw(symbol)
        updated = self.download(symbol, current)
        self.store(symbol, updated)
        return preload(updated)
        
    def preload(self, data):
        return data
        
        
class YfinanceBase(BaseData):
    
    def download(self, symbol, current=None, overlap=5):
        if current is None:
            updated = yf.download(symbol)
            intraday_return = np.log(updated['Close'] - updated['Open'])
            overnight_return = np.log(updated['Close']).diff() - np.log(updated['Adj Close']).diff()
            
            updated.loc[updated.index[-1], ['High', 'Low', 'Close', 'Adj Close', 'Volume']] = np.nan
            
    def preload(self, data):
        return data


class LocalDataStore(BaseData):
    
    pass


class FredBase(BaseData):
    
    pass


class RateBase(BaseData):
    
    pass
    
    
class Yfinance(YfinanceBase, LocalDataStore):
    
    pass
    
class Fred(FredBase, LocalDataStore):

    pass
    
class FredRate(Fred, RateBase):
    
    pass
    
