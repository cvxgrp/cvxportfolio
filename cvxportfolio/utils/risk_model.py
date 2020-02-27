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


def fit_factor_model_risk(returns,
                          num_factors=10,
                          SVDTOL=1E-2,
                          MAX_SVD_ITERS=100,
                          regularizer=1E-8):

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

        # if num_factors < 99:
        u, s, v = spl.svds(normalized, k=num_factors)
#         else:
#             u, s, v = np.linalg.svd(normalized, full_matrices=False)
#             s = s[:num_factors][::-1]
#             u = u[:,:num_factors][:,::-1]
#             #v = v.T
#             v = v[:num_factors][::-1]

        if na_count == 0:
            break

        low_rank_approx = u[:, :] @ np.diag(s[:]) @ v[:, ]

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

    factors = pd.DataFrame(v, columns=returns.columns)
    factors = factors.iloc[::-1].reset_index(drop=True)

    factor_exposures = factors * stds
    factor_covariance = np.diag((s[:])[::-1]**2) / returns.shape[0]

    factor_diag = np.diag(factor_exposures.T @
                          factor_covariance @ factor_exposures)
    idyo_variances = pd.Series(np.maximum((stds**2) - factor_diag, 0.),
                               index=stds.index) + regularizer

    # factor_returns = pd.DataFrame(u[:, ::-1], index=returns.index)

    return factor_exposures, factor_covariance, idyo_variances,  # factor_returns, stds


def factor_covariance_log_lik(factor_exposures,
                              factor_covariance,
                              idyo_variances,
                              returns):

    assert not np.any(np.any(returns.isnull()))

    cov = factor_exposures.T @ factor_covariance @ \
        factor_exposures + np.diag(idyo_variances)

    log_determinant = np.log(np.linalg.eig(cov)[0]).sum()
    cov_inv = np.linalg.inv(cov)

    return (-log_determinant * returns.shape[0] -
            np.trace(cov_inv @ (returns.T @ returns).values))
