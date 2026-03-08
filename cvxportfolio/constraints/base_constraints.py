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
"""This module defines base constraint classes."""


import cvxpy as cp
import numpy as np

from ..errors import ConvexityError
from ..estimator import CvxpyExpressionEstimator, DataEstimator

__all__ = ['Constraint', 'EqualityConstraint', 'InequalityConstraint']

class Constraint(CvxpyExpressionEstimator):
    """Base cvxpy constraint class."""

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :returns: some cvxpy.constraints object, or list of those
        :rtype: cvxpy.constraints, list
        """
        raise NotImplementedError # pragma: no cover


class EqualityConstraint(Constraint):
    """Base class for equality constraints.

    This class is not exposed to the user, each equality
    constraint inherits from this and overrides the
    :func:`InequalityConstraint._compile_constr_to_cvxpy` and
    :func:`InequalityConstraint._rhs` methods.

    We factor this code in order to streamline the
    design of :class:`SoftConstraint` costs.
    """

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :returns: Cvxpy constraints object.
        :rtype: cvxpy.constraints
        """
        return self._compile_constr_to_cvxpy(
            w_plus=w_plus, z=z, w_plus_minus_w_bm=w_plus_minus_w_bm, **kwargs
                ) == self._rhs()

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Cvxpy expression of the left-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover

    def _rhs(self):
        """Cvxpy expression of the right-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover


class InequalityConstraint(Constraint):
    """Base class for inequality constraints.

    This class is not exposed to the user, each inequality
    constraint inherits from this and overrides the
    :func:`InequalityConstraint._compile_constr_to_cvxpy` and
    :func:`InequalityConstraint._rhs` methods.

    We factor this code in order to streamline the
    design of :class:`SoftConstraint` costs.
    """

    def compile_to_cvxpy(
            self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :raises ConvexityError: If the compiled constraint is not convex.

        :returns: Cvxpy constraints object.
        :rtype: cvxpy.constraints
        """

        _ = self._compile_constr_to_cvxpy(
            w_plus=w_plus, z=z, w_plus_minus_w_bm=w_plus_minus_w_bm, **kwargs
                ) <= self._rhs()
        if not _.is_dcp():
            raise ConvexityError(f"The constraint {self} is not convex!")
        assert _.is_dcp(dpp=True)
        return _

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Cvxpy expression of the left-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover

    def _rhs(self):
        """Cvxpy expression of the right-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover


class CostInequalityConstraint(InequalityConstraint):
    """Inequality constraint applied to a cost (risk) term.

    The user does not interact with this class directly.
    It is returned by an expression such as ``cost <= value``
    where ``cost`` is a :class:`Cost` instance and ``value`` is a scalar.

    When the cost implements :meth:`~cvxportfolio.costs.Cost._soc_expression`,
    the constraint is compiled as a native Second-Order Cone (SOC) constraint
    ``cp.norm2(v) <= t``, which is numerically better conditioned than the
    scalar ``cp.sum_squares(v) <= t**2`` fallback used otherwise.
    """

    def __init__(self, cost, value):
        self.cost = cost
        self.value = DataEstimator(
            value, compile_parameter=True, parameter_shape='scalar')
        self._sqrt_value_param = None

    def initialize_estimator(self, **kwargs):
        """Create the sqrt-of-limit parameter used in the SOC constraint.

        :param kwargs: All keyword arguments forwarded from
            :meth:`~cvxportfolio.estimator.Estimator.initialize_estimator_recursive`.
        :type kwargs: dict
        """
        self._sqrt_value_param = cp.Parameter(nonneg=True)

    def values_in_time(self, **kwargs):
        """Update the sqrt-of-limit parameter.

        Called after sub-estimators (including :attr:`value`) have already
        updated their own parameters, so ``self.value.parameter.value`` is
        guaranteed to be current.

        :param kwargs: All keyword arguments forwarded from
            :meth:`~cvxportfolio.estimator.Estimator.values_in_time_recursive`.
        :type kwargs: dict
        """
        self._sqrt_value_param.value = np.sqrt(
            max(float(self.value.parameter.value), 0.0))

    def compile_to_cvxpy( # pylint: disable=arguments-differ
            self, w_plus, z, w_plus_minus_w_bm, **kwargs):
        """Compile constraint to cvxpy, using SOC form when available.

        If the wrapped cost implements
        :meth:`~cvxportfolio.costs.Cost._soc_expression`, the constraint is
        emitted as a native SOC constraint::

            cp.norm2(v) <= sqrt_limit_param

        where ``v`` is the vector returned by ``_soc_expression`` and
        ``sqrt_limit_param`` equals ``sqrt(limit)`` at each trading step.
        Otherwise the scalar fallback ::

            cost_scalar_expr <= limit_param

        is used (identical to the pre-SOC behaviour).

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :param kwargs: Reserved for future expansion.
        :type kwargs: dict

        :raises ConvexityError: If the compiled constraint is not convex.

        :returns: Cvxpy constraint object.
        :rtype: cvxpy.constraints.constraint.Constraint
        """
        soc_vec = self.cost._soc_expression(
            w_plus=w_plus, z=z,
            w_plus_minus_w_bm=w_plus_minus_w_bm, **kwargs)
        if soc_vec is not None:
            result = cp.norm2(soc_vec) <= self._sqrt_value_param
            if not result.is_dcp():
                raise ConvexityError(
                    f"The constraint {self} is not convex!")
            assert result.is_dcp(dpp=True)
            return result
        # Fallback: scalar sum_squares path
        result = self.cost.compile_to_cvxpy(
            w_plus=w_plus, z=z,
            w_plus_minus_w_bm=w_plus_minus_w_bm, **kwargs
        ) <= self.value.parameter
        if not result.is_dcp():
            raise ConvexityError(
                f"The constraint {self} is not convex!")
        assert result.is_dcp(dpp=True)
        return result

    def _compile_constr_to_cvxpy( # pylint: disable=arguments-differ
            self, **kwargs):
        """Scalar-path LHS expression (used by the fallback path)."""
        return self.cost.compile_to_cvxpy(**kwargs)

    def _rhs(self):
        """Scalar-path RHS parameter."""
        return self.value.parameter

    def __repr__(self):
        return self.cost.__repr__() + ' <= ' + self.value.__repr__()
