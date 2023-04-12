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

"""
This module implements Estimator base classes. Policies, costs, and constraints inherit
from this.
"""


class Estimator:
    """Estimator base class.

    Policies, costs, and constraints inherit from this.
    """

    children = []

    def prescient_evaluation(self, returns, volumes, **kwargs):
        """Cache computation with prescience recursively on its children.

        This function is called by Simulator classes before the
        start of a backtest with the full dataset available to the
        simulator. It should be used very carefully, if you
        are not sure do not implement this method. If you do,
        do check that at each point in time only past data
        (with respect to that point) is used and not future data.

        Args:
            returns (pandas.DataFrame): market returns
            volumes (pandas.DataFrame): market volumes
            kwargs (dict): extra data available
        
        Returns:
            children_result (list): result of recursive call on children
        """

        return [
            child.prescient_evaluation(returns, volumes, **kwargs) for child in children
        ]

    def values_in_time(self, t, current_portfolio, **kwargs):
        """Evaluates estimator at a point in time recursively on its children.

        This function is called by Simulator classes on Policy classes
        returning the current trades list. Policy classes, if
        they contain internal estimators, should register them
        in `self.children`  and call this base function before
        they do their internal computation. CvxpyExpression estimators
        should instead define this method to update their Cvxpy parameters.

        Args:
            t (pd.TimeStamp): point in time of the simulation
            current_portfolio (pd.Series): current portfolio in value form (e.g., US dollars)
            kwargs (dict): extra data used
        
        Returns:
            children_result (list): result of recursive call on children
        """
        return [
            child.values_in_time(t, current_portfolio, **kwargs) for child in children
        ]


class CvxpyExpression(Estimator):
    """Base class for estimators that are Cvxpy expressions."""

    def compile_to_cvxpy(self, w_plus, z):
        """Compile term to cvxpy expression.

        This is called by a Policy class on its terms before the start of the backtest
        to compile its Cvxpy problem. If the Policy changes in time
        this is called at every time step.

        It can either return a scalar expression, in the case of objective terms,
        or a list of cvxpy constraints, in the case of constraints.

        In MultiPeriodOptimization policies this is called separately
        for costs and constraints at different look-ahead steps with
        the corresponding w_plus and z.

        Args:
            w_plus (cvxpy.Variable): post-trade allocation weights vector
            z (cvxpy.Variable): trades weight vector

        Returns:
            cvxpy.Expression
        """
        raise NotImplementedError


class DataEstimator(Estimator):
    """Estimator of point-in-time values from internal `self.data`."""

    pass
