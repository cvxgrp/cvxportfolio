# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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

import pandas as pd
import numpy as np

__all__ = ['periods_per_year', 'resample_returns', 'flatten_heterogeneous_list']

def periods_per_year(idx):
    """Given a datetime pandas index return the periods per year."""
    return int(np.round(len(idx) / ((idx[-1] - idx[0]) / pd.Timedelta('365.24d'))))
    
def resample_returns(returns, periods):
    """Resample returns expressed over number of periods to single period."""
    return np.exp(np.log(1 + returns) / periods) - 1

def flatten_heterogeneous_list(l):
    """[1, 2, 3, [4, 5]] -> [1, 2, 3, 4, 5]"""
    return sum(([el] if not hasattr(el, '__iter__') 
        else el for el in l), [])