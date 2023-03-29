"""
Copyright 2017 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

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

import pytest
import numpy as np

from cvxportfolio import values_in_time, null_checker

def test_null_checker(returns):
    """Test null check."""
    with pytest.raises(ValueError):
        null_checker(np.NaN)

    assert null_checker(1.) is None
    assert null_checker(returns) is None

    returns.iloc[0, 0] = np.NaN
    with pytest.raises(ValueError):
        null_checker(returns)

def test_time_locator(returns):
    """Test time locator."""
    t = returns.index[10]
    assert np.allclose(returns.loc[t], values_in_time(returns, t))
    assert np.allclose(23, values_in_time(23, t))
    assert np.allclose(returns.loc[t], values_in_time(returns.loc[t], t))
