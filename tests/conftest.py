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

"""global fixtures"""
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def sigma(resource_dir):
    return pd.read_csv(resource_dir / "sigmas.csv", index_col=0, parse_dates=[0])


@pytest.fixture()
def returns(resource_dir):
    return pd.read_csv(resource_dir / "returns.csv", index_col=0, parse_dates=[0])


@pytest.fixture()
def volumes(resource_dir):
    return pd.read_csv(resource_dir / "volumes.csv", index_col=0, parse_dates=[0])
