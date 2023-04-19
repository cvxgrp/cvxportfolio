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

import copy

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from cvxportfolio import (
    FullCovariance,
    HcostModel,
    MultipleReturnsForecasts,
    LegacyReturnsForecast,
    SinglePeriodOpt,
    TcostModel,
    simulator,
)


@pytest.fixture()
def tcost_model(volumes, sigma):
    return TcostModel(volumes, sigma, 0.0005, 1.0)


@pytest.fixture()
def hcost_model():
    return HcostModel(0.0, 0.0)


def test_attribution(returns, volumes, sigma, tcost_model, hcost_model):
    """Test attribution."""
    # Alpha source
    alpha_sources = [LegacyReturnsForecast(returns, name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
    emp_Sigma = np.cov(returns.to_numpy().T)
    risk_model = FullCovariance(emp_Sigma, gamma=100.0)

    pol = SinglePeriodOpt(
        alpha_model, [
            risk_model, tcost_model, hcost_model], [], solver=cvx.ECOS)

    tcost = TcostModel(0.0005, 1.0, sigma, volumes)
    hcost = HcostModel(0.0)
    market_sim = simulator.MarketSimulator(
        returns, costs=[tcost, hcost], market_volumes=volumes
    )

    p_0 = pd.Series(index=returns.columns, data=1e6)
    noisy = market_sim.run_backtest(
        p_0, returns.index[1], returns.index[10], pol)
    # linear fit attribution
    attr = market_sim.attribute(noisy, pol, parallel=False, fit="linear")
    base_line = noisy.v - sum(p_0)
    for i in range(3):
        assert np.allclose(
            attr[i] / weights[i] / sum(p_0),
            base_line / sum(p_0))

    assert np.allclose(attr["RMS error"], np.zeros(len(noisy.v)))

    # least-squares fit attribution
    attr = market_sim.attribute(
        noisy, pol, parallel=False, fit="least-squares")
    base_line = noisy.v - sum(p_0)
    for i in range(3):
        assert np.allclose(
            attr[i] / weights[i] / sum(p_0),
            base_line / sum(p_0))

    # Residual always 0.
    alpha_sources = [
        LegacyReturnsForecast(
            returns * 0,
            name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
    pol.alpha_model = alpha_model
    attr = market_sim.attribute(
        noisy, pol, parallel=False, fit="least-squares")
    assert np.allclose(attr["residual"], np.zeros(len(noisy.v)))


def test_attribute_non_profit_series(
        returns,
        sigma,
        volumes,
        tcost_model,
        hcost_model):
    """Test attributing series quantities besides profit."""
    # Alpha source
    alpha_sources = [LegacyReturnsForecast(returns, name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
    emp_Sigma = np.cov(returns.to_numpy().T)
    risk_model = FullCovariance(emp_Sigma, gamma=100.0)
    # tcost_model = TcostModel(self.a, self.b, sigma, volumes)
    # hcost_model = HcostModel(self.s, self.s * 0)
    pol = SinglePeriodOpt(
        alpha_model, [
            risk_model, tcost_model, hcost_model], [], solver=cvx.ECOS)

    tcost = TcostModel(0.0005, 1.0, sigma, volumes)

    # tcost = TcostModel(volumes, sigma, self.a, self.b)
    hcost = HcostModel(0.0)
    market_sim = simulator.MarketSimulator(
        returns, costs=[tcost, hcost], market_volumes=volumes
    )

    p_0 = pd.Series(index=returns.columns, data=1e6)
    noisy = market_sim.run_backtest(
        p_0, returns.index[1], returns.index[10], pol)
    # Select tcosts.

    def selector(result):
        return result.leverage

    # linear fit attribution
    attr = market_sim.attribute(
        noisy,
        pol,
        selector,
        parallel=False,
        fit="linear")
    base_line = noisy.leverage
    for i in range(3):
        np.allclose(attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))

    np.allclose(attr["RMS error"], np.zeros(len(noisy.v)))

    # least-squares fit attribution
    attr = market_sim.attribute(
        noisy, pol, selector, parallel=False, fit="least-squares"
    )
    for i in range(3):
        np.allclose(attr[i] / weights[i] / sum(p_0), base_line / sum(p_0))
    # Residual always 0.
    alpha_sources = [
        LegacyReturnsForecast(
            returns * 0,
            name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
    pol = copy.copy(pol)
    pol.alpha_model = alpha_model
    attr = market_sim.attribute(
        noisy, pol, selector, parallel=False, fit="least-squares"
    )
    assert np.allclose(attr["residual"], np.zeros(len(noisy.v)))


def test_attribute_non_profit_scalar(returns, sigma, volumes):
    """Test attributing scalar quantities besides profit."""
    # Alpha source
    alpha_sources = [LegacyReturnsForecast(returns, name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)
    emp_Sigma = np.cov(returns.to_numpy().T)
    risk_model = FullCovariance(emp_Sigma)
    tcost_model = TcostModel(0.0005, 1.0, sigma, volumes)

    # tcost = TcostModel(volumes, sigma, self.a, self.b)
    hcost_model = HcostModel(0.0)
    pol = SinglePeriodOpt(
        alpha_model, [
            100 * risk_model, tcost_model, hcost_model], [])

    market_sim = simulator.MarketSimulator(
        returns, costs=[tcost_model, hcost_model])

    p_0 = pd.Series(index=returns.columns, data=1e6)
    noisy = market_sim.run_backtest(
        p_0, returns.index[1], returns.index[10], pol)
    # Select tcosts.

    def selector(result):
        return pd.Series(index=[noisy.h.index[-1]], data=result.volatility)

    # linear fit attribution
    attr = market_sim.attribute(
        noisy,
        pol,
        selector,
        parallel=False,
        fit="linear")
    base_line = noisy.volatility
    for i in range(3):
        assert np.allclose(
            attr[i][0] / weights[i] / sum(p_0),
            base_line / sum(p_0))
    assert np.allclose(attr["RMS error"], np.zeros(len(noisy.v)))

    # least-squares fit attribution
    attr = market_sim.attribute(
        noisy, pol, selector, parallel=False, fit="least-squares"
    )
    for i in range(3):
        assert np.allclose(
            attr[i][0] / weights[i] / sum(p_0),
            base_line / sum(p_0))
    # Residual always 0.
    alpha_sources = [
        LegacyReturnsForecast(
            returns * 0,
            name=i) for i in range(3)]
    weights = np.array([0.1, 0.3, 0.6])
    alpha_model = MultipleReturnsForecasts(alpha_sources, weights)

    pol.alpha_model = alpha_model
    attr = market_sim.attribute(
        noisy, pol, selector, parallel=False, fit="least-squares"
    )
    assert np.allclose(attr["residual"], np.zeros(len(noisy.v)))