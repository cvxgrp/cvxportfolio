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
"""This module defines cost models for both simulation and optimization.

We implement two types of costs, as discussed in the paper:
:class:`TransactionCost`, defined in :paper:`section 2.3 <section.2.3>`, and
:class:`HoldingCost`, defined in :paper:`section 2.4 <section.2.4>`. We also
provide versions of each that are specialized to the stock market,
:class:`StocksTransactionCost` and :class:`StocksHoldingCost`: these have a
default values that are typical for liquid stocks in the US.

The latter two are included by default in
:class:`cvxportfolio.StockMarketSimulator`, the market simulator specialized to
the stock market, which is used throughout the :doc:`examples <examples>`. So,
the example back-tests all include these costs, unless otherwise specified.

The cost objects have the same user interface when you use them as simulator
costs or optimization costs. For example, to include an annualized
borrowing cost of 10% (as opposed to the default 5%) in a back-test, you do

.. code-block:: python

    simulator = cvx.MarketSimulator(
        universe, costs=[cvx.HoldingCost(short_fees=10)])
    backtest_result_with_borrow_fee = simulator.backtest(policy)

And to include the same borrow cost penalty in an optimization-based policy,
you do

.. code-block:: python

    policy = cvx.SinglePeriodOptimization(
        objective = cvx.ReturnsForecast()
            - 0.5 * cvx.FullCovariance()
            - cvx.HoldingCost(short_fees=10),
        constraints = [cvx.LeverageLimit(3)])

As done throughout the library, you can pass data in the form of a Python
scalar (like 10), a Pandas Series indexed by time, by the assets, or a
Pandas DataFrame indexed by time and with the assets as columns; see the
:ref:`manual page on passing data <passing-data>`.
"""

import copy
import warnings
from numbers import Number

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints.base_constraints import (CostInequalityConstraint,
                                           EqualityConstraint,
                                           InequalityConstraint)
from .errors import ConvexSpecificationError
from .estimator import (CvxpyExpressionEstimator, DataEstimator, Estimator,
                        SimulatorEstimator)
from .forecast import HistoricalMeanVolume, HistoricalStandardDeviation
from .hyperparameters import HyperParameter, _resolve_hyperpar
from .utils import periods_per_year_from_datetime_index, resample_returns

__all__ = ["HoldingCost", "TransactionCost", "SoftConstraint",
           "StocksTransactionCost", "StocksHoldingCost", "TcostModel",
           "HcostModel"]


class Cost(CvxpyExpressionEstimator): # pylint: disable=abstract-method
    """Base class for cost objects (and also risks).

    You should derive from this class to define an objective term for
    optimization-based policies, like a risk model.

    The base class itself defines logic used for the algebraic operations, and
    to take inequalities of a cost object (which results in a Cvxportfolio
    :doc:`constraint <constraints>` object).
    """

    def __mul__(self, other):
        """Multiply by constant."""
        if isinstance(other, (Number, HyperParameter)):
            return MulCost(scalar=other, cost=self)
        return NotImplemented

    def __add__(self, other):
        """Add cost expression to another cost expression."""
        if isinstance(other, Cost):
            return SumCost(self, other)
        return NotImplemented

    def __rmul__(self, other):
        """Multiply by constant."""
        return self * other

    def __radd__(self, other):
        """Add cost expression to another cost."""
        return self + other

    def __neg__(self):
        """Take negative of cost."""
        return self * -1

    def __sub__(self, other):
        """Subtract other cost."""
        return self.__add__(-other)

    def __rsub__(self, other):
        """Subtract from other cost."""
        return other.__add__(-self)

    def __le__(self, other):
        """Self <= other, return CostInequalityConstraint."""
        # TODO: if series, we should check that it is dt indexed
        # (DataEstimator should?)
        if isinstance(other, (Number, pd.Series)):
            return CostInequalityConstraint(self, other)
        return NotImplemented

    def __lt__(self, other):
        """Self < other."""
        raise SyntaxError(
            'Strict inequalities are not allowed in convex programs.')

    def __gt__(self, other):
        """Self > other."""
        raise SyntaxError(
            'Strict inequalities are not allowed in convex programs.')

    def __ge__(self, other):
        """Self >= other, return CostInequalityConstraint."""
        return (-self).__le__(-other)

    def copy_keeping_multipliers(self):
        """This method is used when creating MPO policies.

        We want to deepcopy the constituent cost objects, but not the
        multipliers (which can be symbolic HPs).

        :returns: Same cost object, copied.
        :rtype: cvx.costs.Cost
        """
        # TODO: Not finished (it's not copying the HPs inside the costs).
        return copy.deepcopy(self)

class SumCost(Cost):
    """Sum of two cost objects.

    This should not be instatiated directly, use cost + cost.
    """

    def __init__(self, left, right):
        self.left = left
        assert isinstance(left, Cost)
        self.right = right
        assert isinstance(right, Cost)

    def compile_to_cvxpy(self, *args, **kwargs):
        """Compile cost by iterating over constituent costs.

        :param args: Symbolic variables
        :type args: tuple
        :param kwargs: Symbolic variables
        :type kwargs: dict

        :raises ConvexSpecificationError: If there are issues with the convex
            rules of the combined cost.

        :returns: Symbolic expression of the combined cost.
        :rtype: cvxpy.Expression
        """
        s = self.left.compile_to_cvxpy(
            *args, **kwargs) + self.right.compile_to_cvxpy(*args, **kwargs)
        if not s.is_dcp():
            raise ConvexSpecificationError(self)
        return s

    def __repr__(self):
        """Pretty print."""
        ri = str(self.right)
        if ri[0] == '-':
            return str(self.left) + ' ' + ri
        return str(self.left) + ' + ' + ri

    def copy_keeping_multipliers(self):
        """This method is used when creating MPO policies.

        We want to deepcopy the constituent cost objects, but not the
        multipliers (which can be symbolic HPs).

        :returns: Deep copy of the same object, with copy of reference to
            same multipliers.
        :rtype: cvx.costs.Cost
        """
        return self.left.copy_keeping_multipliers(
            ) + self.right.copy_keeping_multipliers()

class MulCost(Cost):
    """Multiplication of scalar and cost.

    This should not be instatiated directly, use scalar * cost.
    """

    def __init__(self, scalar, cost):
        self.scalar = scalar
        assert isinstance(scalar, (Number, HyperParameter))
        self.cost = cost
        assert isinstance(cost, Cost)

    def compile_to_cvxpy(self, *args, **kwargs):
        """Compile cost by iterating over constituent costs.

        :param args: Symbolic variables
        :type args: tuple
        :param kwargs: Symbolic variables
        :type kwargs: dict

        :raises ConvexSpecificationError: If there are issues with convexity
            of the combined cost.

        :returns: Symbolic expression of the combined cost.
        :rtype: cvxpy.Expression
        """
        mul = _resolve_hyperpar(
            self.scalar) * self.cost.compile_to_cvxpy(*args, **kwargs)
        if not mul.is_dcp():
            raise ConvexSpecificationError(self) # pragma: no cover
        assert mul.is_dcp(dpp=True)
        return mul

    def __repr__(self):
        """Pretty print."""
        add_parenthesis = isinstance(self.cost, SumCost)
        result = '- ' if (self.scalar == -1) else (str(self.scalar) + ' * ')
        result += ('(' if add_parenthesis else '') + str(self.cost)
        return result + (')' if add_parenthesis else '')

    def copy_keeping_multipliers(self):
        """This method is used when creating MPO policies.

        We want to deepcopy the constituent cost objects, but not the
        multipliers (which can be symbolic HPs).

        :returns: Deep copy of the same cost, with copy of reference to same
            multipliers.
        :rtype: cvx.costs.Cost
        """
        return MulCost(
            scalar=self.scalar, cost=self.cost.copy_keeping_multipliers())


class SoftConstraint(Cost):
    """Soft constraint cost.

    This can be applied to most :doc:`constraint objects <constraints>`,
    as discussed in :ref:`its section of the constraints documentation
    <soft-constraints>`.

    :param constraint: Cvxportfolio constraint instance whose violation
        we penalize.
    :type constraint: cvxportfolio.constraints.EqualityConstraint or
        cvxportfolio.constraints.InequalityConstraint
    """

    def __init__(self, constraint):
        self.constraint = constraint

    def compile_to_cvxpy( # pylint: disable=inconsistent-return-statements
        self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :raises SyntaxError: If the constraint is not a EqualityConstraint or
             InequalityConstraint.

        :return: Cvxpy expression of the soft constraint.
        :rtype: cvxpy.Expression
        """

        try:
            expr = (self.constraint._compile_constr_to_cvxpy(
                w_plus, z, w_plus_minus_w_bm) - self.constraint._rhs())

        except AttributeError as exc:
            raise SyntaxError(
                f"{self.__class__.__name__} can only be used with"
                " EqualityConstraint or InequalityConstraint instances."
                    ) from exc

        if isinstance(self.constraint, EqualityConstraint):
            return cp.sum(cp.abs(expr))
        if isinstance(self.constraint, InequalityConstraint):
            return cp.sum(cp.pos(expr))


def _annual_percent_to_per_period(value, ppy):
    """Transform annual percent to per-period return.

    :param value: Annual percent return.
    :type value: float
    :param ppy: Periods per year.
    :type ppy: int

    :returns: Per-period return.
    :rtype: float
    """
    return resample_returns(returns=value/100, periods=ppy)


class SimulatorCost( # pylint: disable=abstract-method
    SimulatorEstimator, Cost):
    """Cost class that can be used by :class:`cvxportfolio.MarketSimulator`.

    This is the base class of both :class:`HoldingCost` and
    :class:`TransactionCost`.

    This class derives from :class:`Cost` and
    :class:`cvxportfolio.estimator.SimulatorEstimator`.
    It implements the :meth:`simulate` method (which is abstract in
    :class:`cvxportfolio.estimator.SimulatorEstimator`). The implementation
    uses the CVXPY compiled expression of the (optimization) cost to evaluate
    the cost in simulation, so we're sure the algebra is (exactly) the same.
    Of course the CVXPY expression operates on weights, so holdings and trades
    are divided by the portfolio value, and the result is multiplied by the
    portfolio value. If you implement a custom simulator's cost and prefer
    to implement the cost expression directly you can derive straight from
    :class:`cvxportfolio.estimator.SimulatorEstimator`. Look at the
    code of the cost classes we implement if in doubt. Every operation that
    is different in simulation and in optimization is wrapped in a
    :class:`cvxportfolio.estimator.SimulatorEstimator` which implements
    different :meth:`values_in_time` and :meth:`simulate` respectively.
    """

    def initialize_estimator( # pylint: disable=arguments-differ
        self, universe, **kwargs):
        """Initialize cost by compiling its CVXPY expression (if applies).

        This must be called by derived classes if you want to use
        an internal CVXPY expression to evaluate the simulator cost.

        :param universe: Current trading universe.
        :type universe: pd.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, 'compile_to_cvxpy'):
            self._w_plus = cp.Variable(len(universe))
            self._z = cp.Variable(len(universe))
            self._w_plus_minus_w_bm = cp.Variable(len(universe))
            self._cvxpy_expression = self.compile_to_cvxpy(
                w_plus = self._w_plus, z = self._z,
                w_plus_minus_w_bm = self._w_plus_minus_w_bm)

    def simulate( # pylint: disable=arguments-differ,too-many-arguments
        self, t, u, h_plus, past_volumes,
        past_returns, current_prices,
        current_weights, current_portfolio_value, **kwargs):
        """Simulate the cost in the market simulator (not optimization).

        Cost classes that are meant to be used in the simulator can
        implement this. The arguments to this are the same as for
        :meth:`cvxportfolio.estimator.Estimator.values_in_time` plus the
        trades vector and post-trade allocations.

        :param t: Current timestamp.
        :type t: pandas.Timestamp
        :param u: Trade vector in cash units requested by the policy.
            If the market simulator implements rounding by number of shares
            and/or canceling trades on assets whose volume for the period
            is zero, this is after those transformations.
        :type u: pandas.Series
        :param h_plus: Post-trade holdings vector.
        :type h_plus: pandas.Series
        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param past_volumes: Past market volumes, or None if not available.
        :type past_volumes: pandas.DataFrame or None
        :param current_prices: Current (open) prices, or None if not available.
        :type current_prices: pandas.Series or None
        :param current_weights: Current allocation weights (before trading).
        :type current_weights: pandas.Series
        :param current_portfolio_value: Current total value of the portfolio
            in cash units, before costs.
        :type current_portfolio_value: float
        :param kwargs: Other unused arguments to
            :meth:`cvxportfolio.estimator.SimulatorEstimator.simulate`.
        :type kwargs: dict

        :returns: Simulated realized value of the cost, in cash accounting
            units (e.g., US dollars). Typically a positive number: it is
            subtracted from the cash account.
        :rtype: float
        """

        self.values_in_time(
            t=t, past_volumes=past_volumes, past_returns=past_returns,
            current_prices=current_prices, current_weights=current_weights,
            current_portfolio_value=current_portfolio_value)

        self._w_plus.value = h_plus.values / current_portfolio_value
        # TODO: w_plus_minus_w_bm is unaccounted for (should be?)
        self._z.value = u.values / current_portfolio_value
        return self._cvxpy_expression.value * current_portfolio_value


class YearDividedByTradingPeriod(SimulatorEstimator):
    """Length of a year divided by this trading period's.

    This is used by :class:`HoldingCost` to model its separate behaviors in
    optimization and simulation.

    :param periods_per_year: If provided, overrides internal estimation of
        number of periods per year in optimization. Has no effect in
        simulation. Default None.
    :type periods_per_year: int or None
    """

    def __init__(self, periods_per_year=None):
        self.periods_per_year = periods_per_year

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Evaluate in optimization.

        :param past_returns: Past market returns, we use its index to estimate
            the typical length of a trading period.
        :type past_returns: pd.DataFrame
        :param kwargs: Other unused arguments to
            :meth:`Estimator.values_in_time`.
        :type kwargs: dict

        :returns: Trading periods per year.
        :rtype: float
        """
        if self.periods_per_year is None:
            return periods_per_year_from_datetime_index(past_returns.index)
        return self.periods_per_year

    def simulate( # pylint: disable=arguments-differ
            self, t, t_next, **kwargs):
        """Evaluate in simulation.

        We use the real time length of the trading period, which we know.

        :param t: Current timestamp.
        :type t: pd.Timestamp
        :param t_next: Timestamp of next trading period.
        :type t_next: pd.Timestamp
        :param kwargs: Other unused arguments to
            :meth:`SimulatorEstimator.simulate`.
        :type kwargs: dict

        :returns: Trading periods per year.
        :rtype: float
        """
        return pd.Timedelta('365.24d') / (t_next - t)


class HoldingCost(SimulatorCost):
    r"""Generic holding cost model.

    This is a generalization of the model described in :paper:`section 2.4
    <section.2.4>` of the paper (which instead corresponds to
    :class:`StocksHoldingCost`).

    This represents the following objective term,
    expressed in terms of the post-trade dollar positions:

    .. math::

        s^T_t {(h^+_t)}_- + l^T_t {(h^+_t)}_+ - d^T_t h^+_t

    where :math:`s_t` are the (short) borrowing fees,
    :math:`l_t` are the fees on long positions,
    and :math:`d_t` are dividend rates (their sign is flipped because
    the costs are deducted from the cash account at each period).

    :param short_fees: Short borrowing fees expressed as annual percentage;
        you can provide them as a float (constant for all times and all
        assets), a :class:`pd.Series` indexed by time (constant for all
        assets but varying in time) or by assets' names (constant in time
        but varying across assets), or a :class:`pd.DataFrame` indexed by
        time and whose columns are the assets' names, if varying both
        in time and across assets. See :ref:`the manual page on passing data
        <passing-data>`. If `None` (the default) the term is
        ignored.
    :type short_fees: float, pd.Series, pd.DataFrame or None
    :param long_fees: Fees on long positions expressed as annual percentage;
        same convention as above applies.
    :type long_fees: float, pd.Series, pd.DataFrame or None
    :param dividends: Dividend rates per period. Same conventions as above.
        Our default data interface already includes the dividend rates in the
        market returns (*i.e.*, uses total returns, from the adjusted
        prices). If, however, you provide your own market returns that do not
        include dividends, you may use this. Default None, which disables the
        term.
    :type dividends: float, pd.Series, pd.DataFrame or None
    :param periods_per_year: Number of trading period per year, used to obtain
        the holding cost per-period from the annualized percentages. Only
        relevant in optimization, since in simulation we know the exact
        duration of each period. If left to the default, None, uses the
        estimated average length of each period from the historical data.
        Note that, in simulation, the holding cost is applied to the actual
        length of the period between trading times, so for example it will
        be higher over a weekend than between weekdays.
    :type periods_per_year: float or None
    """

    def __init__(self, short_fees=None, long_fees=None, dividends=None,
                 periods_per_year=None):

        self.short_fees = None if short_fees is None else DataEstimator(
            short_fees)
        self.long_fees = None if long_fees is None else DataEstimator(
            long_fees)
        self.dividends = None if dividends is None else DataEstimator(
            dividends)
        self.periods_per_year = YearDividedByTradingPeriod(periods_per_year)
        self._short_fees_parameter = None
        self._long_fees_parameter = None
        self._dividends_parameter = None

    def initialize_estimator(self, universe, **kwargs):
        """Initialize cvxpy parameters.

        We don't use the parameter from
        :class:`cvxportfolio.estimator.DataEstimator` because we need to
        divide the value by periods_per_year.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """

        if self.short_fees is not None:
            self._short_fees_parameter = cp.Parameter(len(universe) - 1,
                nonneg=True)

        if self.long_fees is not None:
            self._long_fees_parameter = cp.Parameter(len(universe) - 1,
                nonneg=True)

        if self.dividends is not None:
            self._dividends_parameter = cp.Parameter(len(universe) - 1)

        # Used by SimulatorCost
        super().initialize_estimator(universe=universe, **kwargs)

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Update cvxpy parameters.

        We compute the estimate of periods per year from past returns
        (if not provided by the user) and populate the cvxpy parameters
        with the current values of the user-provided data, transformed
        to per-period.

        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        if self.short_fees is not None:
            self._short_fees_parameter.value = np.ones(
                self._short_fees_parameter.size
                ) * _annual_percent_to_per_period(
                    self.short_fees.current_value,
                    self.periods_per_year.current_value)

        if self.long_fees is not None:
            self._long_fees_parameter.value = np.ones(
                self._long_fees_parameter.size
                ) * _annual_percent_to_per_period(
                    self.long_fees.current_value,
                    self.periods_per_year.current_value)

        if self.dividends is not None:
            self._dividends_parameter.value =\
                np.ones(self._dividends_parameter.size
                ) * self.dividends.current_value

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression.
        :rtype: cvxpy.expression
        """

        expression = 0.

        if self.short_fees is not None:
            expression += self._short_fees_parameter.T @ cp.neg(w_plus[:-1])

        if self.long_fees is not None:
            expression += self._long_fees_parameter.T @ cp.pos(w_plus[:-1])

        if self.dividends is not None:
            # we have a minus sign because costs are deducted from PnL
            expression -= self._dividends_parameter.T @ w_plus[:-1]

        assert expression.is_convex()
        return expression


class StocksHoldingCost(HoldingCost):
    r"""Holding cost specialized to stocks (only borrow fee).

    This implements the simple model described in :paper:`section 2.4
    <section.2.4>` of the paper, refer to that for more details. The cost is,
    in terms of the post-trade dollar positions:

    .. math::

        s^T_t {(h^+_t)}_-,

    *i.e.*, a simple borrow fee applies to the short positions. This class is a
    simplified version of :class:`HoldingCost`: it drops the ``long_fees`` and
    ``dividends`` parameters and keeps only ``short_fees`` with a default
    value of 5, *i.e.*, :math:`5\%` annualized borrowing fee. That is a rough
    (optimistic) approximation of the cost of shorting liquid US stocks. This
    cost is included **by default** in :class:`StockMarketSimulator`, the
    market simulator specialized to US (liquid) stocks.

    :param short_fees: Same as in :class:`HoldingCost`: annualized borrow fee
        in percent, can be asset- and/or period-specific. Default 5.
    :type short_fees: float, pd.Series, pd.DataFrame or None
    """

    def __init__(self, short_fees=5):
        super().__init__(short_fees=short_fees)


class VolumeHatOrRealized(SimulatorEstimator):
    r"""Predictor of market volumes used by :class:`TransactionCost`.

    This is used to model the different behaviors in optimization and
    simulation. (In the latter case we use the realized volumes.)

    :param volume_hat: Estimator of the :math:`\hat V` in optimization. Unused
        in simulation.
    :type volume_hat: cvx.estimator.Estimator
    """

    def __init__(self, volume_hat):
        self.volume_hat = volume_hat

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Evaluate in optimization.

        :param kwargs: All arguments to
            :meth:`estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :returns: Current estimate of market volumes.
        :rtype: np.ndarray
        """
        return self.volume_hat.current_value

    def simulate( # pylint: disable=arguments-differ
            self, current_volumes, **kwargs):
        """Evaluate in simulation.

        :param current_volumes: Current market volumes.
        :type current_volumes: pd.Series
        :param kwargs: All other arguments to
            :meth:`estimator.SimulatorEstimator.simulate`.
        :type kwargs: dict

        :raises SyntaxError: If the market volumes are not present in the
            market data.

        :returns: Current market volumes.
        :rtype: np.ndarray
        """
        if not self.volume_hat.current_value is None:
            raise SyntaxError(
                'You should not pass realized volumes to the volume_hat'
                ' argument of TransactionCost; in simulation the volumes are'
                ' provided by the market data server.')
        if current_volumes is None:
            raise SyntaxError(
                "If you don't provide volumes you should set b to None"
                " in the market simulator's TransactionCost object.")
        return current_volumes.values

class TransactionCost(SimulatorCost):
    r"""This is a generic model for transaction cost of financial assets.

    .. versionadded:: 1.2.0

        This was significantly improved with new options; it was undocumented
        before.

    It is described in :paper:`section 2.3 <section.2.3>` of the paper.

    The model is, separated on a single asset (equation 2.2 in the paper)

    .. math ::

        a | x | + b  \sigma \frac{{ | x |}^{3/2}}{V^{1/2}} + c x

    where :math:`x` is the dollar traded quantity,
    :math:`a` is a coefficient representing fees proportional to the absolute
    value traded, like half the bid-ask spread,
    :math:`b` is a coefficient that multiplies the market impact term,
    typically of the order of 1,
    :math:`\sigma` is an estimate of the volatility of the asset returns over
    recent periods,
    :math:`V` is the market volume traded over the period for the asset
    and :math:`c` is a coefficient used to introduce bias in the model,
    for example the negative of open-to-close return (if transactions are
    executed at close), or the negative of the open-to-VWAP return (if
    transactions are executed at the volume-weighted average price).

    In optimization the realized market volume :math:`V` is not known and
    we use its forecast :math:`\hat V` instead.

    As done throughout the library, this implementation accepts either
    :ref:`user-provided data <passing-data>` for the various parts of the
    model, or uses built-in :doc:`forecaster classes <forecasts>` to do the
    heavy-lifting.

    :param a: Coefficients of the first term of the transaction cost model, for
        example half the bid-ask spread, brokerage fees proportional to the
        size of each trade, .... :ref:`Usual conventions on passing data
        <passing-data>` apply. Default None, which disables the term.
    :type a: float, pd.Series, pd.DataFrame, or None
    :param b: Coefficients of the second term of the transaction cost model,
        typically of the order of 1. Same conventions. Default None, which
        disables the term and invalidates the following three parameters.
    :type b: float, pd.Series, pd.DataFrame, or None
    :param volume_hat: Forecast of the market volumes, has only effect in
        optimization (in simulation the actual volumes are used). You can
        pass a DataFrame of externally computed forecasts, or use the default
        :class:`cvxportfolio.forecast.HistoricalMeanVolume` (or another
        forecaster) to compute the forecasts at each point in time. If you pass
        a class, like the default, it is instantiated with parameter
        ``rolling`` equal to 1 year, if you prefer a different estimation
        lenght you can instantiate the forecaster and pass the instance.
    :type volume_hat: float, pd.Series, pd.DataFrame, cvx.forecast.BaseForecast
        class or instance
    :param sigma: Externally computed forecasts, or forecaster, of the market
        returns' volatilities :math:`\sigma`. The default is
        :class:`cvxportfolio.forecast.HistoricalStandardDeviation`. If you
        pass a class, like the default, it is instantiated with parameter
        ``rolling`` equal to 1 year, if you prefer a different estimation
        lenght you can instantiate the forecaster and pass the instance.
    :type sigma: float, pd.Series, pd.DataFrame, cvx.forecast.BaseForecast
        class or instance
    :param exponent: Exponent of the second term of the model, default
        :math:`3/2`. (In the model above, this exponent applies to
        :math:`|z|`, and this exponent minus 1 applies to the denominator
        term :math:`V`). You can use any float larger than 1.
    :type exponent: float
    :param c: Coefficients of the third term of the transaction cost model.
        If None, the default, the term is ignored.
    :type c: float, pd.Series, pd.DataFrame, or None
    """

    def __init__( # pylint: disable=too-many-arguments
        self, a=0., b=None, volume_hat=HistoricalMeanVolume,
        sigma=HistoricalStandardDeviation, exponent=1.5, c=None):

        self.a = None if a is None else DataEstimator(a)
        self.b = None if b is None else DataEstimator(b)
        self.c = None if c is None else DataEstimator(
            c, compile_parameter=True)

        if self.b is not None:
            if isinstance(volume_hat, type):
                volume_hat = volume_hat(
                    rolling=pd.Timedelta('365.24d'))
            self.market_volumes = VolumeHatOrRealized(
                DataEstimator(volume_hat))

            if isinstance(sigma, type):
                sigma = sigma(
                    rolling=pd.Timedelta('365.24d'))
            self.sigma = DataEstimator(sigma)

        if exponent < 1.:
            raise SyntaxError(
                'Exponent should be >=1, otherwise the'
                ' transaction cost model is not convex.')
        self.exponent = exponent
        self._first_term_multiplier = None
        self._second_term_multiplier = None

    def initialize_estimator(self, universe, **kwargs):
        """Initialize cvxpy parameters.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        if self.a is not None:
            self._first_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)
        if self.b is not None:
            self._second_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)

        # SimulatorCost
        super().initialize_estimator(universe=universe, **kwargs)

    def values_in_time( # pylint: disable=arguments-differ
            self, current_portfolio_value, **kwargs):
        """Update cvxpy parameters.

        :param current_portfolio_value: Current total value of the portfolio.
        :type current_portfolio_value: float
        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """
        if self.a is not None:
            self._first_term_multiplier.value = np.ones(
                self._first_term_multiplier.size) * self.a.current_value

        if self.b is not None:

            self._second_term_multiplier.value =\
                (self.b.current_value * self.sigma.current_value
                ) / ((self.market_volumes.current_value + 1E-8)
                    / current_portfolio_value) ** (self.exponent - 1)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression.
        :rtype: cvxpy.expression
        """
        expression = 0
        if self.a is not None:
            expression += cp.abs(z[:-1]) @ self._first_term_multiplier
            assert expression.is_convex()
        if self.b is not None:
            expression += (
                cp.abs(z[:-1]) ** (self.exponent)
                    ) @ self._second_term_multiplier
            assert expression.is_convex()
        if self.c is not None:
            expression += cp.sum(z[:-1] * self.c.parameter)
        return expression

# Backward compatibility before 1.2.0

class SimpleSigmaEst(SimulatorEstimator):
    """Simple estimator of sigma for backward compatibility."""

    def __init__(self, window_sigma_est):
        warnings.warn(
            "Passing a number to window_sigma_est is deprecated, "
            "You should use a forecaster like the default "
            "HistoricalStandardDeviation, instantiating with"
            " its 'rolling' argument to choose the length of estimation.",
            DeprecationWarning)
        self.window_sigma_est = window_sigma_est

    def values_in_time(
        # pylint: disable=arguments-differ
        self, past_returns, **kwargs):
        """Compute historical sigma.

        :param past_returns: Past market returns.
        :type past_returns: pd.DataFrame
        :param kwargs: Other unused arguments.
        :type kwargs: dict

        :returns: Estimated sigma
        :rtype: np.array
        """

        return np.sqrt(
            (past_returns.iloc[-self.window_sigma_est:, :-1]**2).mean()).values

    def simulate(
        # pylint: disable=arguments-differ
        self, current_returns, past_returns, **kwargs):
        """Compute historical sigma.

        :param current_returns: Current market returns.
        :type current_returns: pd.Series
        :param past_returns: Past market returns.
        :type past_returns: pd.DataFrame
        :param kwargs: Other unused arguments.
        :type kwargs: dict

        :returns: Estimated sigma
        :rtype: np.array
        """

        return np.std(pd.concat(
            [past_returns.iloc[-self.window_sigma_est + 1:, :-1],
            pd.DataFrame(current_returns.iloc[:-1]).T], axis=0), axis=0).values

        # sum = (past_returns.iloc[-self.window_sigma_est-1:, :-1]**2).sum()
        # count = past_returns.iloc[-self.window_sigma_est-1:, :-1].count()
        # sum += current_returns**2
        # count += ~current_returns.isnull()
        # return np.sqrt(sum / count).values

# Backward compatibility before 1.2.0

class SimpleVolumeEst(Estimator):
    """Simple estimator of volume for backward compatibility."""

    def __init__(self, window_volume_est):
        warnings.warn(
            "Passing a number to window_volume_est is deprecated, "
            "You should use a forecaster like the default "
            "HistoricalMeanVolume, instantiating with"
            " its 'rolling' argument to choose the length of estimation.",
            DeprecationWarning)

        self.window_volume_est = window_volume_est

    def values_in_time(
        # pylint: disable=arguments-differ
        self, past_volumes, **kwargs):
        """Compute historical mean of volumes.

        :param past_volumes: Past market volumes
        :type past_volumes: pd.DataFrame
        :param kwargs: Other unused arguments.
        :type kwargs: dict

        :raises SyntaxError: If the market data does not contain volumes.

        :returns: Estimated volume
        :rtype: np.array
        """

        if past_volumes is None:
            raise SyntaxError(
                "If you don't provide market volumes you can not use the"
                " market impact term of the transaction cost model.")

        return past_volumes.iloc[-self.window_volume_est:].mean().values

class StocksTransactionCost(TransactionCost):
    """Simplified version of :class:`TransactionCost` for stocks.

    .. versionadded:: 1.2.0

        We added the ``sigma`` and ``volume_hat`` parameters to support
        user-provided values as well as forecasters with various parameters.
        We deprecated the old way (``window_sigma_est`` and
        ``window_volume_est``) in which that was done before.

    This is included as a simulator cost by default (with default arguments) in
    :class:`cvxportfolio.StockMarketSimulator`.

    :param a: Same as in :class:`TransactionCost`, default 0.
    :type a: float or pd.Series or pd.DataFrame
    :param pershare_cost: Per-share cost: cash paid for each share traded.
        Requires to know the prices of the stocks (they are present in the
        default market data server
        :class:`cvxportfolio.data.DownloadedMarketData`). Default 0.005.
    :type pershare_cost: float or pd.Series or pd.DataFrame
    :param b: Same as in :class:`TransactionCost`, default 1.
    :type b: float or pd.Series or pd.DataFrame
    :param sigma: Same as in :class:`TransactionCost`.
    :type sigma: float, pd.Series, pd.DataFrame, cvx.forecast.BaseForecast
        class or instance
    :param window_sigma_est: Deprecated, size of rolling window to
        estimate historical standard deviations. If left to None (the default)
        has no effect. Use instead the ``sigma`` parameter.
    :type window_sigma_est: int or None
    :param volume_hat: Same as in :class:`TransactionCost`.
    :type volume_hat: float, pd.Series, pd.DataFrame, cvx.forecast.BaseForecast
        class or instance
    :param window_volume_est: Deprecated, size of rolling window to estimate
        the mean of past volumes. If left to None (the default)
        has no effect. Use instead the ``volume_hat`` parameter.
    :type window_volume_est: int or None
    :param exponent: Same as in :class:`TransactionCost`.
    :type exponent: float
    :param c: Same as in :class:`TransactionCost`.
    :type c: float or pd.Series or pd.DataFrame or None
    """

    def __init__( # pylint: disable=too-many-arguments
        self, a=0., pershare_cost=0.005, b=1.0,
        volume_hat=HistoricalMeanVolume, sigma=HistoricalStandardDeviation,
        exponent=1.5, c=None, window_sigma_est=None, window_volume_est=None):

        if window_sigma_est is not None:
            sigma = SimpleSigmaEst(window_sigma_est)

        if window_volume_est is not None:
            volume_hat = SimpleVolumeEst(window_volume_est)

        super().__init__(# because we update it with pershare_cost
                         a= 0. if a is None else a,
                         b=b, c=c, exponent=exponent,
                         volume_hat=volume_hat,
                         sigma=sigma,
                         )

        self.pershare_cost = DataEstimator(pershare_cost)\
            if pershare_cost is not None else None

    def values_in_time( # pylint: disable=arguments-renamed
            self, current_prices, **kwargs):
        """Update linear cost with per-share cost.

        :param current_prices: Current (open) market prices.
        :type current_prices: pd.Series
        :param kwargs: Other unused arguments to
            :meth:`cvxportfolio.estimator.Estimator.values_in_time`.
        :type kwargs: dict

        :raises SyntaxError: If market prices are not available and
            pershare_cost is not None.
        """

        super().values_in_time(current_prices=current_prices, **kwargs)

        if self.pershare_cost is not None:
            if current_prices is None:
                raise SyntaxError("If the market data doesn't contain prices"
                                  " you should set pershare_cost to None")
            assert not np.any(current_prices.isnull())
            assert not np.any(current_prices == 0.)
            self._first_term_multiplier.value += \
                self.pershare_cost.current_value / current_prices.values

# Aliases

class TcostModel(TransactionCost):
    """Alias of :class:`TransactionCost`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """

class HcostModel(HoldingCost):
    """Alias of :class:`HoldingCost`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """
