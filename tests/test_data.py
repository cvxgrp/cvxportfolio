"""
Copyright 2023- The Cvxportfolio Contributors

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

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio.data import YfinanceBase



def test_yfinance_download():
    """Test YfinanceBase."""

    data = YfinanceBase().download('AAPL')
    print(data)
    print(data.loc['2023-04-10']['Return'])
    print(data.loc['2023-04-11', 'Open'] / data.loc['2023-04-10', 'Open'] - 1)
    assert np.isclose(data.loc['2023-04-10', 'Return'], data.loc['2023-04-11', 'Open'] / data.loc['2023-04-10', 'Open'] - 1)
