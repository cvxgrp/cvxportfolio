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
"""This module contains classes to make forecasts such as historical means
and covariances and are used internally by cvxportfolio objects. In addition,
forecast classes have the ability to cache results online so that if multiple
classes need access to the estimated value (as is the case in MultiPeriodOptimization
policies) the expensive evaluation is only done once. 
"""

import numpy as np

from .estimator import Estimator


class BaseForecast(Estimator):
    """Base class for forecasters."""
    
    # def pre_evaluation(self, universe, backtest_times):
    #     self.universe = universe
    #     self.backtest_times = backtest_times
    
    def update_chooser(self, t, past_returns):
        if (self.last_time is None) or (self.last_time != past_returns.index[-1]):
            self.compute_from_scratch(t=t, past_returns=past_returns)
            # print('FROM SCRATCH!', t)
        else:
            # print('UPDATING!', t)
            self.update(t=t, past_returns=past_returns)
    
    def compute_from_scratch(self, t, past_returns):
        raise NotImplementedError

    def update(self, t, past_returns):
        raise NotImplementedError

class HistoricalMeanReturn(BaseForecast):
    """Historical mean returns."""
    
    def __init__(self, lastforcash):
        self.lastforcash = lastforcash
        self.last_time = None
        self.last_counts = None
        self.last_sum = None
    
    def values_in_time(self, t, past_returns, **kwargs):
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
        
        self.update_chooser(t=t, past_returns=past_returns)
            
        self.current_value = (self.last_sum / self.last_counts).values
        # self.current_value = past_returns.mean().values
        
        if self.lastforcash:
            self.current_value[-1] = past_returns.iloc[-1, -1]
        
        return self.current_value
        
    def compute_from_scratch(self, t, past_returns):
        self.last_counts = past_returns.count()
        self.last_sum = past_returns.sum()
        self.last_time = t
        
    def update(self, t, past_returns): #, last_estimation, last_counts, last_time):
        self.last_counts += ~(past_returns.iloc[-1].isnull())
        self.last_sum += past_returns.iloc[-1].fillna(0.)
        self.last_time = t

        # if last_time is None: # full estimation
        #     estimation = past_returns.sum()
        #     counts = past_returns.count()
        # else:
        #     assert last_time == past_returns.index[-2]
        #     estimation = last_estimation * last_counts + past_returns.iloc[-1].fillna(0.)
        #     counts = last_counts + past_returns.iloc[-1:].count()
        #
        # return estimation/counts, counts, past_returns.index[-1]
        
        
class HistoricalMeanError(BaseForecast):
    """Historical standard deviations of the mean."""

    def __init__(self):#, zeroforcash):
        # self.zeroforcash = zeroforcash
        # assert zeroforcash=True
        self.varianceforecaster = HistoricalVariance(addmean=False)
    
    def values_in_time(self, t, past_returns, **kwargs):
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
                
        self.current_value  = np.sqrt(self.varianceforecaster.current_value / self.varianceforecaster.last_counts.values)
        # if self.zeroforcash:
        #     self.current_value[-1] = 0.
        return self.current_value  
        
        
class HistoricalVariance(BaseForecast):
    """Historical variances."""

    def __init__(self, addmean):
        self.addmean = addmean
        if not self.addmean:
            self.meanforecaster = HistoricalMeanReturn(lastforcash=False)
        self.last_time = None
        self.last_counts = None
        self.last_sum = None
    
    def values_in_time(self, t, past_returns, **kwargs):
        super().values_in_time(t=t, past_returns=past_returns.iloc[:,:-1], **kwargs)
        
        self.update_chooser(t=t, past_returns=past_returns.iloc[:,:-1])
        
        self.current_value = (self.last_sum / self.last_counts).values
                
        if not self.addmean:
            self.current_value -= self.meanforecaster.current_value**2

        return self.current_value  
        
    def compute_from_scratch(self, t, past_returns):
        self.last_counts = past_returns.count()
        self.last_sum = (past_returns**2).sum()
        self.last_time = t
        
    def update(self, t, past_returns): #, last_estimation, last_counts, last_time):
        self.last_counts += ~(past_returns.iloc[-1].isnull())
        self.last_sum += past_returns.iloc[-1].fillna(0.)**2
        self.last_time = t
        
          
class HistoricalFactorizedCovariance(BaseForecast):
    """Historical covariance matrix, sqrt factorized."""
    
    def __init__(self, addmean=True):
        #assert addmean == True
        self.addmean = addmean
        # if not self.addmean:
        #     self.meanforecaster = HistoricalMeanReturn(lastforcash=False)
        self.last_time = None
    
    def get_count_matrix(self, past_returns):
        """We obtain the matrix of non-null joint counts."""
        tmp = (~past_returns.isnull()) * 1.
        return tmp.T @ tmp

    @staticmethod
    def factorize(Sigma):
        eigval, eigvec = np.linalg.eigh(Sigma)
        eigval = np.maximum(eigval, 0.)
        return eigvec @ np.diag(np.sqrt(eigval))
        
    def compute_from_scratch(self, t, past_returns):
        self.last_counts_matrix = self.get_count_matrix(past_returns).values
        filled = past_returns.fillna(0.).values
        self.last_sum_matrix = filled.T @ filled
        # if not self.addmean:
        #     self.last_meansum_matrix = self.last_sum_matrix/self.last_counts_matrix - past_returns.cov(ddof=0)
        #     self.last_meansum_matrix *= self.last_counts_matrix**2
        self.last_time = t
        
    def update(self, t, past_returns): #, last_estimation, last_counts, last_time):
        nonnull = ~(past_returns.iloc[-1].isnull())
        self.last_counts_matrix += np.outer(nonnull, nonnull)
        last_ret = past_returns.iloc[-1].fillna(0.)
        self.last_sum_matrix += np.outer(last_ret, last_ret)
        self.last_time = t
        # if not self.addmean:
        #     self.last_meansum_matrix += np.outer(last_ret, last_ret)
    
    
    def values_in_time(self, t, past_returns, **kwargs):
        super().values_in_time(t=t, past_returns=past_returns.iloc[:, :-1], **kwargs)
        if self.addmean:
            self.update_chooser(t=t, past_returns=past_returns.iloc[:,:-1])
            Sigma = self.last_sum_matrix / self.last_counts_matrix
        else:
            Sigma = past_returns.iloc[:,:-1].cov(ddof=0)
        # if not self.addmean:
        #     Sigma -= np.outer(self.meanforecaster.current_value, self.meanforecaster.current_value)   
        self.current_value = self.factorize(Sigma) 
        
        return self.current_value