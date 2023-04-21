# Copyright 2023- The Cvxportfolio Contributors
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

from cvxportfolio.simulator import MarketSimulator


def test_simulator_raises():

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator()

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator(returns=pd.DataFrame([[0.]]))

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator(volumes=pd.DataFrame([[0.]]))

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator(returns=pd.DataFrame(
            [[0.]]), volumes=pd.DataFrame([[0.]]))

    # not raises
    simulator = MarketSimulator(returns=pd.DataFrame([[0., 0.]]), volumes=pd.DataFrame(
        [[0.]]), per_share_fixed_cost=0., round_trades=False)

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator(returns=pd.DataFrame(
            [[0., 0.]]), volumes=pd.DataFrame([[0.]]), per_share_fixed_cost=0.)

    with pytest.raises(SyntaxError):
        simulator = MarketSimulator(returns=pd.DataFrame(
            [[0., 0.]]), volumes=pd.DataFrame([[0.]]), round_trades=False)


def test_prepare_data(tmp_path):
    simulator = MarketSimulator(['ZM', 'META'], base_location=tmp_path)
    assert simulator.returns.data.shape[1] == 3
    assert simulator.prices.data.shape[1] == 2
    assert simulator.volumes.data.shape[1] == 2
    assert simulator.sigma_estimate.data.shape[1] == 2
    assert np.isnan(simulator.returns.data.iloc[-1, 0])
    assert np.isnan(simulator.volumes.data.iloc[-1, 1])
    assert not np.isnan(simulator.prices.data.iloc[-1, 0])
    assert simulator.returns.data.index[-1] == simulator.volumes.data.index[-1]
    assert simulator.returns.data.index[-1] == simulator.prices.data.index[-1]
    assert simulator.sigma_estimate.data.index[-1] == simulator.prices.data.index[-1]
    assert np.isclose(simulator.sigma_estimate.data.iloc[-1,0],
         simulator.returns.data.iloc[-1001:-1,0].std())
         
def test_methods(tmp_path):
    simulator = MarketSimulator(['ZM', 'META', 'AAPL'], base_location=tmp_path)
    super(simulator.__class__, simulator).values_in_time('2023-04-14', None, None, None, None)
    
    for i in range(10):
        np.random.seed(i)
        tmp = np.random.uniform(size=4)*1000
        tmp[3] = -sum(tmp[:3])
        u = pd.Series(tmp, simulator.returns.data.columns)
        rounded = simulator.round_trade_vector(u)
        assert sum(rounded) == 0
        assert np.linalg.norm(rounded[:-1] - u[:-1]) < \
            np.linalg.norm(simulator.prices.data.loc['2023-04-14']/2)
        
        print(u)
    
    # raise Exception
