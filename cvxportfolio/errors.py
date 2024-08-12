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
"""This module defines some Exceptions thrown by Cvxportfolio objects."""

__all__ = ['DataError', 'UserDataError', 'MissingTimesError',
           'NaNError', 'MissingAssetsError', 'ForecastError',
           'PortfolioOptimizationError',
           'ProgramInfeasible', 'ProgramUnbounded',
           'Bankruptcy', 'ConvexSpecificationError', 'ConvexityError']


class DataError(ValueError):
    """Base class for exception related to data."""


class UserDataError(DataError, SyntaxError):
    """Exception for errors in data provided by the user."""


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


class NumericalSolverError(PortfolioOptimizationError):
    """Numerical solver failed to produce a solution."""


class ProgramInfeasible(PortfolioOptimizationError):
    """Optimization program is infeasible."""


class ProgramUnbounded(PortfolioOptimizationError):
    """Optimization program is unbounded."""


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
