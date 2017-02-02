"""
Copyright 2017 Enzo Busseti.

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

def factor_model_covariance(returns, variance_explained=.75):
    """Compute factor model given a returns dataframe.
    
    Derived from the procedure in 
    https://faculty.washington.edu/ezivot/research/factormodellecture_handout.pdf
    
    The risk model is given by 
    $$ F^T \Sigma_{factor} F + \diag(\sigma^2_{idyo}) $$
    
    Args:
        - returns: DataFrame of dimension T x N
        - variance_explained: float in [0,1], how much empirical variance 
                              we want the factor model to explain
    
    Returns:
        - factor_returns: DataFrame of dimension T x n_fact
        - sigma_factor: DataFrame of dimension n_fact x n_fact
        - F: DataFrame of dimension n_fact x N
        - sigma2_idyo: Series of dimension N
    """
    T,N = returns.shape
    returns_demeaned=returns-returns.mean()
    u,s,v = np.linalg.svd(returns_demeaned)
    nfact = ((s**2/sum(s**2)).cumsum()>=variance_explained).nonzero()[0][0]
    factor_returns = pd.DataFrame(index=returns.index, data=u[:, :nfact])
    factor_sigma_sqrt = pd.DataFrame(np.diag(s[:nfact]))
    F = pd.DataFrame(columns=returns.columns, data=v[:nfact, :])
    explained_returns = pd.DataFrame(index=returns.index, columns=returns.columns,
                 data=factor_returns.values @ factor_sigma_sqrt.values @ F.values)
    
    sigma2_idyo = (returns_demeaned - explained_returns).var()
    sigma_factor = factor_sigma_sqrt**2/(T-1)

    return factor_returns, sigma_factor, F, sigma2_idyo

def series_factor_model_covariance(returns, variance_explained=.75, freq='MS',
                                    lookback_periods = 24):
    period_starts = pd.date_range(start=returns.index[0],
                        end=returns.index[-1], freq='MS')

    exposures, factor_sigma, idyos = {}, {},{}
    for i in range(lookback_periods, len(period_starts)):
        t = period_starts[i]
        used_returns = returns.loc[(returns.index < period_starts[i])&
                        (returns.index >= period_starts[i-lookback_periods])]
        factor_returns, sigma_factor, F, sigma2_idyo = factor_model_covariance(used_returns,
                                                                variance_explained=variance_explained)
        exposures[t]=F
        factor_sigma[t]=sigma_factor
        idyos[t]=sigma2_idyo
    
    return pd.Panel(exposures).fillna(value=0), pd.Panel(factor_sigma).fillna(value=0), pd.DataFrame(idyos).T

if __name__ == '__main__':
    data = pd.read_csv('../../returns.txt',index_col=0,parse_dates=[0])
    returns=data
    for variance_explained in [.5,.75,.8,.9]:
        factor_returns, factor_sigma, exposures, idyos = factor_model_covariance(returns, variance_explained=variance_explained)
        fullcov=np.cov(returns.T)
        reconst = exposures.T.values @ factor_sigma.values @ exposures.values + np.diag(idyos)
        print('%.2f var expl: max |dev| %.2e, mean |dev| %.2e' % (variance_explained,
                np.max(np.abs(fullcov - reconst)), np.mean(np.abs(fullcov - reconst))))
    exposures_pan, factor_sigma_pan, idyos_df = \
        series_factor_model_covariance(returns, variance_explained=.75, freq='MS',
                                        lookback_periods = 24)
