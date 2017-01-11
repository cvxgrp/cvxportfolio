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
import pandas as pd

# TODO - generalize to n dimensions & write test
# TODO - add flag to return signal scaled with real sigma and real mu

def _generate_ou_process(len_proc, th = .1, mu = 0, x0 = 0, std = 1):
    """Base generator (has numerical instabilities for big len_proc)"""
    sig = std*np.sqrt(2*th);
    t = np.linspace(0, len_proc, len_proc)
    ex = np.exp(-th*t);
    return x0*ex+mu*(1-ex)+ (1/np.sqrt(2*th))*\
      sig*ex*np.concatenate([[0],np.cumsum(np.sqrt(np.diff(np.exp(2*th*t)-1))*np.random.randn(len(t)-1))])

def generate_ou_process(len_proc, th = .1, mu = 0, x0 = 0, std = 1):
    """Generate OU process
    Args:
        len_proc : lenght of the process
        th : decay speed, .1 has half life of 2 weeks if daily
        mu : mean
        x0 : initial val
        std : stdev of the resulting proc.
    Returns:
        1d array of the process
    """
    proc = _generate_ou_process(len_proc, th = th, mu = mu, x0 = x0, std = std)
    if np.isnan(proc).any():
        return np.concatenate([generate_ou_process(int(len_proc/2), th = th, mu = mu, x0 = x0, std = std),
                               generate_ou_process(len_proc - int(len_proc/2), th = th, mu = mu, x0 = x0, std = std)])
    else:
        return proc

def _generate_alpha_signal(real_returns, IC, seed = None, type_noise = 'white', **kwargs):
    """Given real returns, generates alpha signal with given information coefficient.

    The alpha signal has mean zero and unit variance.

        Args:
            real_returns: iterable, 1 dimensional
            IC: information coefficient of the signal
            type: "white" for white noise
                  "OU" for OU process (autocorrelated noise)

    (Grinold and Kahn, Chapt. 6)"""

    real_returns = np.array(real_returns).flatten()

    if seed is not None:
        np.random.seed(seed)

    target_covariance = np.matrix([[1, IC],
                                   [IC, 1]])
    chol_dec = np.linalg.cholesky(target_covariance)

    if type_noise == 'white':
        noise = np.random.randn(len(real_returns))
    elif type_noise == 'OU':
        noise = generate_ou_process(len(real_returns), **kwargs)
    else:
        raise SyntaxError('Wrong noise type')
    return_normalized = (real_returns - np.mean(real_returns)) / np.std(real_returns)

    return np.dot(chol_dec, np.vstack((return_normalized, noise)))[1,:].A1


def generate_alpha_signal(real_returns, IC, seed=None, type_noise='white', **kwargs):
    """Given real returns, generates alpha signal with given information coefficient.

    The alpha signal has mean zero and unit variance.

        Args:
            real_returns: pandas datframe or pandas series.
                          if DF, the time dimension MUST be the index (for OU noise)
            IC: information coefficient of the signal
            type_noise: "white" for white noise
                  "OU" for OU process (autocorrelated noise)

    (Grinold and Kahn, Chapt. 6)"""
    if isinstance(real_returns, pd.Series):
        index = real_returns.index
        result = _generate_alpha_signal(real_returns.values, IC, seed = seed, **kwargs)
        return pd.Series(index=index, data=result)

    elif isinstance(real_returns, pd.DataFrame):
        index = real_returns.index
        columns = real_returns.columns
        data = real_returns.values.reshape(len(index)*len(columns), order = 'F')  # Time dimension is index
        result = _generate_alpha_signal(data, IC, seed=seed, type_noise=type_noise, **kwargs)
        return pd.DataFrame(index=index, columns=columns,
                            data=result.reshape((len(index), len(columns)), order = 'F'))  # Time dimension is index
    else:
        raise SyntaxError('Only pandas series or dataframes')

if __name__ == "__main__":
    import pandas as pd
    np.random.seed(0)
    returns = pd.DataFrame(index=pd.date_range('2015-01-01', '2015-01-30'),
                           columns=['aaa', 'bbb'],
                        data=0.01*np.random.randn(30,2))
    print('returns', returns)
    alpha_sig = generate_alpha_signal(returns, IC=0.9)
    print('alpha sig', alpha_sig)
    print('corr coeff\n', np.corrcoef(returns.values.flatten(), alpha_sig.values.flatten()))
