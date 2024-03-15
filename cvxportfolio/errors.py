# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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
"""This module defines some Exceptions thrown by Cvxportfolio objects."""

__all__ = ['DataError', 'MissingTimesError',
           'NaNError', 'MissingAssetsError', 'ForecastError',
           'PortfolioOptimizationError', 'Bankruptcy',
           'ConvexSpecificationError', 'ConvexityError']


class DataError(ValueError):
    """Base class for exception related to data."""


class MissingTimesError(DataError):
    """Cvxportfolio couldn't find data for a certain time."""


class NaNError(DataError):
    """Cvxportfolio tried to access data that includes np.nan."""


class MissingAssetsError(DataError):
    """Cvxportfolio couldn't find data for certain assets."""


class ForecastError(DataError):
    """Forecast procedure failed."""


class PortfolioOptimizationError(Exception):
    """Errors with portfolio optimization problems."""


class Bankruptcy(Exception):
    """A backtest resulted in a bankruptcy."""


class ConvexSpecificationError(SyntaxError):
    """Some custom term does not comply with disciplined convex programming."""
    def __init__(self, term):
        super().__init__(
            "The convex optimization term"
            f" {term} does not follow the convex optimization specifications."
            " This could be due to a mis-specified custom term,"
            " or a non-convex cost inequality used as constraint"
            " (e.g., `[cvx.FullCovariance() >= constant]`)."
        )


class ConvexityError(ConvexSpecificationError):
    """Some program term is not convex."""
    def __init__(self, cost):
        super().__init__(
            f"The cost term {cost}"
            " is not concave. (We need concavity since we maximize it.)"
            " You probably have a cost's multiplier with the wrong sign.")
