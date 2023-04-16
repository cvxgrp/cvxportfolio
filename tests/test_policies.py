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

from cvxportfolio.policies import Hold, RankAndLongShort

def test_hold():
    hold = Hold()
    w = pd.Series(.5, ['AAPL', 'CASH'])
    assert np.all(hold.values_in_time(None, w, 'extra_args', hello='ciao').values == np.zeros(2))


def test_rank_and_long_short():
    hold = Hold()
    w = pd.Series(.25, ['AAPL', 'TSLA', 'GOOGL', 'CASH'])
    signal = pd.Series([1,2,3], ['AAPL', 'TSLA', 'GOOGL'])
    num_long = 1
    num_short = 1
    target_leverage = 3.
    rls = RankAndLongShort(signal = signal, num_long = num_long, num_short = num_short, target_leverage=target_leverage)
    z = rls.values_in_time(None, w, 'extra_arg', hello='ciao')
    print(z)
    wplus = w + z
    assert wplus['CASH'] == 1
    assert wplus['TSLA'] == 0
    assert wplus['AAPL'] == -wplus['GOOGL']
    assert np.abs(wplus[:-1]).sum() == 3
    
    index = pd.date_range('2020-01-01', '2020-01-03')
    signal = pd.DataFrame({'AAPL': pd.Series([1,1.9,3], index), 'TSLA': pd.Series([3,2.1,1], index), 'GOOGL': pd.Series([4,4,4], index)})
    rls = RankAndLongShort(signal = signal, num_long = num_long, num_short = num_short, target_leverage=target_leverage)
    z1 = rls.values_in_time(index[0], w, 'extra_arg', hello='ciao')
    print(z1)
    wplus = w + z1
    assert wplus['CASH'] == 1
    assert wplus['TSLA'] == 0
    assert wplus['AAPL'] == -wplus['GOOGL']
    assert np.abs(wplus[:-1]).sum() == 3
    z2 = rls.values_in_time(index[1], w, 'extra_arg', 321, hello='ciao')
    print(z2)
    wplus = w + z2
    assert wplus['CASH'] == 1
    assert wplus['TSLA'] == 0
    assert wplus['AAPL'] == -wplus['GOOGL']
    assert np.abs(wplus[:-1]).sum() == 3
    z3 = rls.values_in_time(index[2], w, 'extra_arg', hola=42, hello='ciao')
    wplus = w + z3
    assert wplus['CASH'] == 1
    assert wplus['AAPL'] == 0
    assert wplus['TSLA'] == -wplus['GOOGL']
    assert np.abs(wplus[:-1]).sum() == 3
    print(z3)
    
    #raise Exception