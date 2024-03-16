# Copyright 2023 Enzo Busseti
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
"""This module defines base constraint classes."""


from ..errors import ConvexityError
from ..estimator import CvxpyExpressionEstimator, DataEstimator

__all__ = ['Constraint', 'EqualityConstraint', 'InequalityConstraint']

class Constraint(CvxpyExpressionEstimator):
    """Base cvxpy constraint class."""

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
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

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :returns: Cvxpy constraints object.
        :rtype: cvxpy.constraints
        """
        return self._compile_constr_to_cvxpy(w_plus, z, w_plus_minus_w_bm) ==\
            self._rhs()

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
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

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile constraint to cvxpy.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :returns: Cvxpy constraints object.
        :rtype: cvxpy.constraints
        """
        _ = self._compile_constr_to_cvxpy(w_plus, z, w_plus_minus_w_bm) <=\
            self._rhs()
        if not _.is_dcp():
            raise ConvexityError(f"The constraint {self} is not convex!")
        assert _.is_dcp(dpp=True)
        return _

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Cvxpy expression of the left-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover

    def _rhs(self):
        """Cvxpy expression of the right-hand side of the constraint."""
        raise NotImplementedError # pragma: no cover


class CostInequalityConstraint(InequalityConstraint):
    """Linear inequality constraint applied to a cost term.

    The user does not interact with this class directly,
    it is returned by an expression such as ``cost <= value``
    where ``cost`` is a :class:`Cost` instance and ``value``
    is a scalar.
    """

    def __init__(self, cost, value):
        self.cost = cost
        self.value = DataEstimator(value, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile constraint to cvxpy."""
        return self.cost.compile_to_cvxpy(w_plus, z, w_plus_minus_w_bm)

    def _rhs(self):
        return self.value.parameter

    def __repr__(self):
        return self.cost.__repr__() + ' <= ' + self.value.__repr__()
