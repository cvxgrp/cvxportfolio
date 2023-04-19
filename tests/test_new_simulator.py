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

from cvxportfolio.simulator import NewMarketSimulator

def test_simulator_raises():
    
    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator()
        
    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator(returns=pd.DataFrame([[0.]]))

    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator(volumes=pd.DataFrame([[0.]]))
        
    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator(returns=pd.DataFrame([[0.]]), volumes=pd.DataFrame([[0.]]))
        
    # not raises
    simulator = NewMarketSimulator(returns=pd.DataFrame([[0., 0.]]), volumes=pd.DataFrame([[0.]]), per_share_fixed_cost=0., round_trades=False)
    
    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator(returns=pd.DataFrame([[0., 0.]]), volumes=pd.DataFrame([[0.]]), per_share_fixed_cost=0.)
        
    with pytest.raises(SyntaxError):
        simulator = NewMarketSimulator(returns=pd.DataFrame([[0., 0.]]), volumes=pd.DataFrame([[0.]]), round_trades=False)
        
def test_prepare_data(tmp_path):
    simulator = NewMarketSimulator(['ZM', 'META'], base_location=tmp_path)
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
    assert simulator.sigma_estimate.data.iloc[-1,0] == simulator.returns.data.iloc[-1001:-1, 0].std()
    
    