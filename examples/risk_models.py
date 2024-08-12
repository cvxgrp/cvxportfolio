# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""Test different choices of risk models, which has best performance?

.. note::

    Some of the interfaces used here (custom hyper-parameter objects) are
    experimental and **are not covered by the semantic versioning agreement**,
    meaning that they could change without notice. In general, methods and
    interfaces shown in the examples are public and are protected
    under semantic versioning, but in this example that may not be the case.

.. note::

    The output of this example is currently (Cvxportfolio ``1.0.3``)
    not too easy to read; the ``__repr__`` method of a policy object
    with symbolic hyper-parameters is scheduled for improvement. It
    does work, though.

On the Dow Jones, daily trading from 2016 to today:
- diagonal risk model
- diagonal risk model with risk forecast error
- full covariance
- full covariance with risk forecast error
- ...

We test on a long-only portfolio and use automatic hyper-parameter
optimization to maximize the information ratio, in back-test,
versus the index ETF.
"""

from pprint import pprint

import numpy as np
import pandas as pd

import cvxportfolio as cvx

from .universes import DOW30 as UNIVERSE

# Index
INDEX = 'DIA'

# Times.
START = '2000-01-01'
END = None # today

# Leverage.
LEVERAGE_LIMIT = 1.

# Stock market simulator with default transaction and
# holding cost models.
simulator = cvx.StockMarketSimulator(UNIVERSE + [INDEX])

# Build benchmark.
all_in_index = pd.Series(
    0., index = simulator.market_data.returns.columns)
all_in_index[INDEX] = 1.
benchmark = cvx.FixedWeights(all_in_index)

# Define hyper-parameter objects.
# These will be included in the library in a future release.
class GammaTradeCoarse(cvx.RangeHyperParameter):
    """Transaction cost multiplier, coarse value range."""
    def __init__(self):
        super().__init__(
        values_range=np.arange(1, 11),
        current_value=1.)

class GammaTradeFine(cvx.RangeHyperParameter):
    """Transaction cost multiplier, fine value range."""
    def __init__(self):
        super().__init__(
        values_range=np.linspace(-1., 1., 51),
        current_value=0)

class GammaRiskCoarse(cvx.RangeHyperParameter):
    """Risk term multiplier, coarse value range."""
    def __init__(self):
        super().__init__(
        values_range=np.arange(1, 21),
        current_value=1.)

class GammaRiskFine(cvx.RangeHyperParameter):
    """Risk term multiplier, fine value range."""
    def __init__(self):
        super().__init__(
        values_range=np.linspace(-1., 1., 51),
        current_value=0)

class Kappa(cvx.RangeHyperParameter):
    """Risk forecast error multiplier, fine value range."""
    def __init__(self):
        super().__init__(
        values_range=np.linspace(0., 0.5),
        current_value=0)


# We test these risk models, with symbolic hyper-parameters.
base_risk_models = [cvx.DiagonalCovariance(), cvx.FullCovariance()]
base_risk_models += [
    cvx.FactorModelCovariance(num_factors=num_factors)
    for num_factors in [1, 2, 5, 10]
]

# with hyper-parameters
RISK_MODELS = [
    (GammaRiskCoarse() + GammaRiskFine()) * base_risk_model
    for base_risk_model in base_risk_models
]

# with risk forecast error
RISK_MODELS += [
    (GammaRiskCoarse() + GammaRiskFine()) * (
    base_risk_model + Kappa() * cvx.RiskForecastError())
    for base_risk_model in base_risk_models
]

results = {}

for risk_model in RISK_MODELS:
    print('Testing risk model:')
    print(risk_model)

    results[repr(risk_model)] = {}
    current_result = results[repr(risk_model)]

    # Build policy.
    policy = cvx.SinglePeriodOpt(
        objective = cvx.ReturnsForecast()
            - (GammaTradeCoarse() + GammaTradeFine()
                ) * cvx.StocksTransactionCost()
            - risk_model,
        constraints=[cvx.LongOnly(), cvx.LeverageLimit(LEVERAGE_LIMIT)],
        benchmark=benchmark,
        solver='CLARABEL')

    # Optimize HPs.
    simulator.optimize_hyperparameters(
        policy, start_time=START, end_time=END,
        objective='information_ratio')

    print('Policy with optimized hyper-parameters:')
    print(policy)
    current_result['Optimized policy'] = repr(policy)

    # Run back-test with optimized HPs.
    optimized_result = simulator.backtest(
        policy, start_time=START, end_time=END)

    print('Back-test result with optimized hyper-parameters:')
    print(optimized_result)
    current_result['Optimized result'] = optimized_result

pprint(results)
