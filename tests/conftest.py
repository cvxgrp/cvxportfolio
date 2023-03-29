"""global fixtures"""
from pathlib import Path

import pytest
import pandas as pd

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
