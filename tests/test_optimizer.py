# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
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

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio.costs import HcostModel, TcostModel
from cvxportfolio.policies import SinglePeriodOpt
from cvxportfolio.returns import ReturnsForecast
from cvxportfolio.risks import FullCovariance


@pytest.fixture()
def tcost_model(sigma, volumes):
    return TcostModel(0, 1.0, sigma, volumes, power=2)


@pytest.fixture()
def hcost_model():
    return HcostModel(0.0, 1e-3)


# def test_single_period_opt(returns, sigma, volumes, tcost_model, hcost_model):
#     """Test single period optimizer."""
#     universe = returns.columns
#     times = returns.index
#
#     # Alpha source
#     gamma = 100.0
#     n = len(universe)
#     alpha_model = ReturnsForecast(returns)
#     emp_Sigma = np.cov(returns.iloc[:,:-1].to_numpy().T) + np.eye(n-1) * 1e-3
#     risk_model = FullCovariance(emp_Sigma)
#     risk_model.set_benchmark(pd.Series(0, returns.columns))
#     pol = SinglePeriodOpt(
#         alpha_model, [
#             gamma * risk_model, tcost_model, hcost_model], [], solver=cvx.ECOS)
#     t = times[1]
#     p_0 = pd.Series(index=universe, data=1e6)
#     z = pol.get_trades(p_0, t)
#     assert z.sum() == pytest.approx(0.0, abs=1e-7)
#     # Compare with CP calculation.
#     h = z + p_0
#     rho = 1.0 * sigma.loc[t] * (sum(p_0) / volumes.loc[t])
#     rho = np.hstack([rho, 0])
#     emp_Sigma = np.cov(returns.to_numpy().T) + np.eye(n) * 1e-3
#     A = 2 * gamma * emp_Sigma + 2 * np.diag(rho)
#     s_val = pd.Series(index=returns.columns, data=1e-3)
#     s_val["cash"] = 0.0
#     b = returns.loc[t] + 2 * rho * (p_0 / sum(p_0)) + s_val
#     h0 = np.linalg.solve(A, b)
#     offset = np.linalg.solve(A, np.ones(n))
#     nu = (1 - h0.sum()) / offset.sum()
#     hstar = h0 + nu * offset
#     assert hstar.sum() == pytest.approx(1.0)
#     assert np.allclose(h / sum(p_0), hstar, atol=1e-6)
