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
    
    def load_raw(self, symbols):
        raise NotImplementedError
        
    def load(self, symbols):
        return self.preload(self.load_raw(symbols))
        
    def store(self, symbols, data):
        raise NotImplementedError
        
    def download(self, symbols, current=None):
        raise NotImplementedError
        
    def update_and_load(self, symbols):
        current = self.load_raw(symbols)
        updated = self.download(symbols, current)
        self.store(symbols, updated)
        return preload(updated)
        
    def preload(self, data):
        return data
        
        
class YfinanceBase(BaseData):
    
    def download(self, symbols, current=None, overlap=5):
        if current is None:
            updated = yf.download()
            
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
    
