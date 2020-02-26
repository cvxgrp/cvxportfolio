"""
Copyright 2020 Enzo Busseti.

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

import pandas as pd
import numpy as np
import scipy.sparse.linalg as spl


def fit_factor_model_risk(returns, num_factors=10,
                          SVDTOL=1E-5, MAX_SVD_ITERS=100):

    means = (returns).mean()
    stds = (returns).std()

    if (np.any(stds.isnull())):
        raise(Exception('Some asset has all NaN returns.'))

    na_mask = returns.isnull()
    na_count = na_mask.sum().sum()

    normalized = (returns - means) / stds

#     print((normalized > 5.).sum().sum())
#     print((normalized < -5.).sum().sum())

#     returns.loc[normalized > 5.] = np.nan
#     returns.loc[normalized < -5.] = np.nan

    if na_count > 0:
        normalized.mask(na_mask, other=0., inplace=True)
        error = pd.DataFrame(index=normalized.index,
                             columns=normalized.columns, data=0.)

    for i in range(MAX_SVD_ITERS):

        u, s, v = spl.svds(normalized, k=num_factors + 1)

        low_rank_approx = u[:, 1:] @ np.diag(s[1:] - s[0]) @ v[1:, ]
        # low_rank_approx = u@np.diag(s)@v
        # loadings = s[:num_factors] -s[num_factors]

        if na_count == 0:
            break

        error.mask(na_mask, other=low_rank_approx - normalized,
                   inplace=True)
        error_measure = (error**2).sum().sum() / na_count

        # print(error_measure)

        if error_measure < SVDTOL:
            break

        normalized.mask(na_mask, other=low_rank_approx,
                        inplace=True)

    else:

        raise(Exception('Iterative SVD did not converge. ' +
                        'Try reducing num_factors.'))

    factors = v[1:, :]
    factors = pd.DataFrame(factors, columns=returns.columns)
    factors = factors.iloc[::-1].reset_index(drop=True)

    factor_exposures = factors * stds
    factor_covariance = np.diag((s[1:] - s[0])[::-1]**2) / returns.shape[0]

    factor_diag = np.diag(factor_exposures.T @
                          factor_covariance @ factor_exposures)
    idyo_variances = stds**2 - factor_diag

    factor_returns = pd.DataFrame(u[:, 1:][:, ::-1], index=returns.index)

    if not np.all(idyo_variances > 0):
        raise(Exception('Low-rank algorithm failed. ' +
                        'Try reducing num_factors.'))

    # print(s)

    # confront with random
    # A = np.random.randn(*returns.shape)
    # A = pd.DataFrame(A)
    # A = (A) / A.std()
    # u, new_s, v = spl.svds(A, k=num_factors)
    # print(s[1:] > new_s)

    return {'factor_exposures': factor_exposures,
            'factor_covariance': factor_covariance,
            'idyo_variances': idyo_variances,
            'factor_returns': factor_returns,
            # 'stds': stds
            }
