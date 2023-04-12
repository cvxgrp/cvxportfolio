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
import numpy as np


class BaseData:
    """Base class for Cvxportfolio database interface."""
    
    def load_raw(self, symbol):
        """Load raw data from database."""
        raise NotImplementedError
        
    def load(self, symbol):
        """Load data from database using `self.preload` function to process it."""
        return self.preload(self.load_raw(symbol))
        
    def store(self, symbol, data):
        """Store data in database."""
        raise NotImplementedError
        
    def download(self, symbol, current=None):
        """Download data from external source."""
        raise NotImplementedError
        
    def update_and_load(self, symbol):
        """Update current stored data for symbol and load it."""
        current = self.load_raw(symbol)
        updated = self.download(symbol, current)
        self.store(symbol, updated)
        return preload(updated)
        
    def preload(self, data):
        """Prepare data to serve to the user."""
        return data
        
        
class YfinanceBase(BaseData):
    """Base class for the Yahoo Finance interface.
    
    This should not be used directly unless you know what you're doing.
    """
    
    def download(self, symbol, current=None, overlap=5, **kwargs):
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
        if current is None:
            updated = yf.download(symbol, **kwargs)
            intraday_logreturn = np.log(updated['Close']) - np.log(updated['Open'])
            close_to_close_logreturn = np.log(updated['Adj Close']).diff().shift(-1)
            open_to_open_logreturn = close_to_close_logreturn + intraday_logreturn - intraday_logreturn.shift(-1)
            updated['Return'] = np.exp(open_to_open_logreturn) - 1
            del updated['Adj Close']
            updated.loc[updated.index[-1], ['High', 'Low', 'Close', 'Return', 'Volume']] = np.nan
            return updated
        else:
            raise NotImplementedError
            
    def preload(self, data):
        """Prepare data for use by Cvxportfolio.
        
        We drop the 'Volume' column expressed in number of stocks
        and replace it with 'ValueVolume' which is an estimate
        of the (e.g., US dollar) value of the volume exchanged
        on the day.
        """
        data['ValueVolume'] = data['Volume'] * data['Open']
        del data['Volume']
        return data


class LocalDataStore(BaseData):
    
    def load_raw(self, symbol):
        """Load raw data from local store."""
        raise NotImplementedError
        
    def store(self, symbol, data):
        """Store data locally."""
        raise NotImplementedError


class FredBase(BaseData):
    
    pass


class RateBase(BaseData):
    
    pass
    
    
class Yfinance(YfinanceBase, LocalDataStore):
    
    """ Yahoo Finance data interface using local data store.
    """
    
    def update_and_load(symbol):
        """Update data for symbol and load it."""
        return super().update_and_load(symbol)
    
class Fred(FredBase, LocalDataStore):

    pass
    
class FredRate(Fred, RateBase):
    
    pass
    
