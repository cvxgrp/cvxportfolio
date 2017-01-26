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

def factor_model_covariance(returns, nfact=10):
    """Compute factor model given a returns dataframe."""
    fullcov=np.cov(returns.T)
    u,s,v = np.linalg.svd(fullcov)
    exposures = pd.DataFrame(index=returns.columns,
                            data=u[:, :nfact])
    factor_sigma = pd.DataFrame(np.diag(s[:nfact]))
    reconst = exposures.values @ factor_sigma.values @ exposures.values.T
    idyos=pd.Series(data=np.diag(fullcov) - np.diag(reconst),
                    index = returns.columns)
    assert min(np.linalg.eigh(reconst + np.diag(idyos))[0]) > 0.
    return exposures.T, factor_sigma, idyos

def series_factor_model_covariance(returns, nfact=10, freq='MS',
                                    lookback_periods = 24):
    period_starts = pd.date_range(start=returns.index[0],
                        end=returns.index[-1], freq='MS')
    exposures_pan = pd.Panel(minor_axis=returns.columns,major_axis =range(nfact))
    factor_sigma_pan = pd.Panel(major_axis = range(nfact), minor_axis=range(nfact))
    idyos_df = pd.DataFrame(columns=returns.columns, dtype=np.float64)
    for i in range(lookback_periods, len(period_starts)):
        t = period_starts[i]
        used_returns = returns.loc[(returns.index < period_starts[i])&
                        (returns.index >= period_starts[i-lookback_periods])]
        exposures, factor_sigma, idyos = factor_model_covariance(used_returns,
                                                                nfact=nfact)
        exposures_pan.loc[t] = exposures
        factor_sigma_pan.loc[t] = factor_sigma
        idyos_df.loc[t,:] = idyos
    return exposures_pan, factor_sigma_pan, idyos_df


if __name__ == '__main__':
    data = pd.read_csv('../../returns.txt',index_col=0,parse_dates=[0])
    returns=data
    for nfact in [1,5,10,20,25]:
        exposures, factor_sigma, idyos = factor_model_covariance(returns, nfact=nfact)
        fullcov=np.cov(returns.T)
        reconst = exposures.T.values @ factor_sigma.values @ exposures.values + np.diag(idyos)
        print('%d fact: max |dev| %.2e, mean |dev| %.2e' % (nfact,
                np.max(np.abs(fullcov - reconst)), np.mean(np.abs(fullcov - reconst))))
    exposures_pan, factor_sigma_pan, idyos_df = \
        series_factor_model_covariance(returns, nfact=10, freq='MS',
                                        lookback_periods = 24)
