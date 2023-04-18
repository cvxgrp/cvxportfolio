"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

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
import pytest

from cvxportfolio import HcostModel, MarketSimulator, SimulationResult, TcostModel


@pytest.fixture()
def portfolio(returns):
    return pd.Series(index=returns.columns, data=1e6)


@pytest.fixture()
def hcost_term():
    return HcostModel(0.0)


@pytest.fixture()
def tcost_term(sigma, volumes):
    return TcostModel(0.0005, 1.0, sigma, volumes)


@pytest.fixture()
def simulator(returns, hcost_term, tcost_term, volumes):
    return MarketSimulator(
        returns, costs=[tcost_term, hcost_term], market_volumes=volumes
    )


@pytest.mark.parametrize("value,expected",
                         [(1e4, 28906767.251), (-1e4, 28908611.931)])
def test_propag(returns, portfolio, simulator, value, expected):
    """Test propagation of portfolio."""
    t = returns.index[1]
    results = SimulationResult(
        initial_portfolio=portfolio,
        policy=None,
        cash_key="cash",
        simulator=simulator)
    u1 = pd.Series(index=portfolio.index, data=value)
    u2 = pd.Series(index=portfolio.index, data=[value] * 29)

    pd.testing.assert_series_equal(u1, u2)

    h_next, u = simulator.propagate(portfolio, u=u1, t=t)
    results.log_simulation(
        t=t,
        u=u,
        h_next=h_next,
        risk_free_return=0.0,
        exec_time=0)
    assert results.simulator_TcostModel.sum().sum() == pytest.approx(157.604, abs=1e-3)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(0.0, abs=1e-3)
    assert sum(h_next) == pytest.approx(expected, abs=1e-3)


def test_propag_list(returns, portfolio, simulator):
    pass
    # obsolete tests only covering initialization of u1


def test_hcost_neg(returns, portfolio, tcost_term, volumes):
    """Test hcost function, negative positions."""
    hcost_term = HcostModel(0.0001)
    simulator = MarketSimulator(
        returns, costs=[tcost_term, hcost_term], market_volumes=volumes
    )

    # simulator.hcost_term.borrow_costs += .0001
    t = returns.index[1]
    results = SimulationResult(
        initial_portfolio=portfolio,
        policy=None,
        cash_key="cash",
        simulator=simulator)
    u = pd.Series(index=portfolio.index, data=-2e6)
    h_next, u = simulator.propagate(portfolio, u, t=t)
    results.log_simulation(
        t=t,
        u=u,
        h_next=h_next,
        risk_free_return=0.0,
        exec_time=0)
    assert results.simulator_HcostModel.sum().sum() == pytest.approx(2800.0)
