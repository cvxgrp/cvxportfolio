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
"""This module defines user-facing constraints."""

import cvxpy as cp
import numpy as np

from ..estimator import DataEstimator, Estimator
from ..forecast import (HistoricalFactorizedCovariance,
                        project_on_psd_cone_and_factorize)
from ..policies import MarketBenchmark
from .base_constraints import (Constraint, EqualityConstraint,
                               InequalityConstraint)

__all__ = [
    "LongOnly",
    "LeverageLimit",
    "LongCash",
    "DollarNeutral",
    "FactorNeutral",
    "ParticipationRateLimit",
    "MaxWeights",
    "MinWeights",
    "MaxBenchmarkDeviation",
    "MinBenchmarkDeviation",
    "NoTrade",
    "NoCash",
    "FactorMaxLimit",
    "FactorMinLimit",
    "FactorGrossLimit",
    "FixedFactorLoading",
    "MarketNeutral",
    "MinWeightsAtTimes",
    "MaxWeightsAtTimes",
    "TurnoverLimit",
    "MinCashBalance"
]

class NoCash(EqualityConstraint):
    """Require that the cash balance is zero at each period."""

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus[-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return 0


class MarketNeutral(EqualityConstraint):
    r"""Simple implementation of β- (or market-) neutrality.

    In our notation, this is

    .. math::
        {(w_t^\text{b})}^T \Sigma_t (w_t + z_t) = 0

    The benchmark portfolio weights are given by a Policy object chosen by
    the user.

    .. versionadded:: 1.2.0

        This constraint's interface has been improved: now you can pass
        any policy object as benchmark, and give parameters to the forecaster
        of :math:`\Sigma_t`.

    :param benchmark: Policy object whose target weights at each point in time
        are the benchmark weights we neutralize against. You can pass a class
        or an instance. If you pass a class it is instantiated with default
        parameters. Default is :class:`cvxportfolio.MarketBenchmark`, which are
        weights proportional to the previous year's total traded volumes.
    :type benchmark: cvx.Policy class or instance
    :param kwargs: Optional arguments passed to the initializer
        of :class:`cvxportfolio.forecast.HistoricalFactorizedCovariance`,
        like rolling window or exponential smoothing half life, for the
        estimation of the covariance matrices :math:`\Sigma_t`. Default (no
        other arguments) is to use its default parameters.
    :type kwargs: dict
    """

    def __init__(self, benchmark=MarketBenchmark, **kwargs):

        if isinstance(benchmark, type):
            benchmark = benchmark()
        self.benchmark = benchmark
        self.covariance_forecaster = HistoricalFactorizedCovariance(**kwargs)
        self._market_vector = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize parameter with size of universe.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._market_vector = cp.Parameter(len(universe)-1)

    def values_in_time( # pylint: disable=arguments-differ
            self, past_volumes, **kwargs):
        """Update parameter with current market weights and covariance.

        :param past_volumes: Past market volumes, in units of value.
        :type past_volumes: pandas.DataFrame
        :param kwargs: Unused arguments passed to :meth:`values_in_time`.
        :type kwargs: dict
        """

        factorized_covariance = self.covariance_forecaster.current_value
        bm = self.benchmark.current_value.iloc[:-1]
        self._market_vector.value = np.array(
            factorized_covariance @ (factorized_covariance.T @ bm))

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus[:-1].T @ self._market_vector

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return 0


class TurnoverLimit(InequalityConstraint):
    r"""Turnover limit as a fraction of the portfolio value.

    The turnover is defined as half the :math:`\ell_1`-norm of
    the trade weight vector, without cash. Here we ask that it is smaller
    than some constant:

    .. math::
        \|{(z_t)}_{1:n}\|_1/2 \leq \delta.

    :param delta: We require that the turnover over each trading period
        is smaller than this value. This is either constant, expressed
        as :class:`float`, or changing in time, expressed as a
        :class:`pd.Series` with datetime index.
    :type delta: float or pd.Series
    """

    def __init__(self, delta):
        self.delta = DataEstimator(delta, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return .5 * cp.norm1(z[:-1])

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.delta.parameter


class ParticipationRateLimit(InequalityConstraint):
    """A limit on maximum trades size as a fraction of market volumes.

    :param volumes: per-stock and per-day market volume estimates, or
        constant in time
    :type volumes: pd.Series or pd.DataFrame
    :param max_fraction_of_volumes: max fraction of market volumes that
        we're allowed to trade
    :type max_fraction_of_volumes: float, pd.Series, pd.DataFrame
    """

    def __init__(self, volumes, max_fraction_of_volumes=0.05):
        self.volumes = DataEstimator(volumes)
        self.max_participation_rate = DataEstimator(
            max_fraction_of_volumes)
        self.portfolio_value = cp.Parameter(nonneg=True)

    def initialize_estimator(self, universe, **kwargs):
        """Initialize internal parameter.

        :param universe: Current trading universe.
        :type universe: pd.Index
        :param kwargs: Unused arguments to initialize estimator.
        :type kwargs: dict
        """
        self._parameter = cp.Parameter(len(universe) - 1)

    def values_in_time( # pylint: disable=arguments-differ
            self, current_portfolio_value, **kwargs):
        """Update parameter with current portfolio value.

        :param current_portfolio_value: Current total value of the portfolio.
        :type current_portfolio_value: float
        :param kwargs: Unused arguments passed to :meth:`values_in_time`.
        :type kwargs: dict
        """
        self.portfolio_value.value = current_portfolio_value
        self._parameter.value = (
            self.volumes.current_value
                * self.max_participation_rate.current_value)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return cp.multiply(cp.abs(z[:-1]), self.portfolio_value)

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self._parameter


class LongOnly(InequalityConstraint):
    r"""A long only constraint.

    .. math::

        w_t + z_t \geq 0

    Imposes that at each point in time the post-trade
    weights are non-negative. By default it applies
    to all elements of the post-trade weights vector
    but you can also exclude the cash account (and let
    cash be negative).

    :param applies_to_cash: Whether the long only requirement
        also applies to the cash account.
    :type applies_to_cash: bool
    """

    def __init__(self, applies_to_cash=False):
        self.applies_to_cash = applies_to_cash

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return -(w_plus if self.applies_to_cash else w_plus[:-1])

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return 0


class NoTrade(Constraint):
    """No-trade condition on name on periods(s).

    .. note::

        This constraint object is experimental and its interface might change.

    :param asset: Symbol which can't be traded on given day(s).
    :type asset: str
    :param periods: Timestamps at which the symbol can't be traded.
    :type periods: iterable of pandas.Timestamp
    """

    def __init__(self, asset, periods):
        self.asset = asset
        self.periods = periods
        self._index = None
        self._low = None
        self._high = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize internal parameters.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._index = (universe.get_loc if hasattr(
            universe, 'get_loc') else universe.index)(self.asset)
        self._low = cp.Parameter()
        self._high = cp.Parameter()

    def values_in_time( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Update parameters, if necessary by imposing no-trade.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param kwargs: Unused arguments passed to :meth:`values_in_time`.
        :type kwargs: dict
        """
        if t in self.periods:
            self._low.value = 0.
            self._high.value = 0.
        else:
            self._low.value = -100.
            self._high.value = +100.

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile constraint to cvxpy, return list of two.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark weights.
        :type w_plus_minus_w_bm: cvxpy.Variable
        :returns: Two constraints.
        :rtype: list of cvxpy.constraints
        """
        return [z[self._index] >= self._low,
                z[self._index] <= self._high]


class LeverageLimit(InequalityConstraint):
    r"""Constraints on the leverage of the portfolio.

    In the notation of the book, this is

    .. math::

        \|{(w_t + z_t)}_{1:n}\|_1 \leq L^\text{max},

    where :math:`(w_t + z_t)` are the post-trade weights, and we
    exclude the cash account from the :math:`\ell_1` norm.

    :param limit: Constant or varying in time leverage limit
        :math:`L^\text{max}`. If varying in time it is expressed
        as a :class:`pd.Series` with datetime index.
    :type limit: float or pd.Series
    """

    def __init__(self, limit):
        self.limit = DataEstimator(limit, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return cp.norm(w_plus[:-1], 1)

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.limit.parameter


class MinCashBalance(InequalityConstraint):
    r"""Require that the cash balance is above a threshold.

    In our notation this is

    .. math::

        {(w_t + z_t)}_{n+1} \geq c_\text{min} / v_t,

    where :math:`v_t` is the portfolio value at time :math:`t`.

    :param c_min: The miminimum cash balance required, either constant in time
        or varying, expressed in units of cash (*e.g.*, US dollars),
        not weight. Use a float if you want a constant limit, or a time-indexed
        Pandas series if you want a limit that changes in time.
    :type c_min: float or pd.Series
    """

    def __init__(self, c_min):
        self.c_min = DataEstimator(c_min)
        self.rhs = cp.Parameter()

    def values_in_time( # pylint: disable=arguments-differ
            self, current_portfolio_value, **kwargs):
        """Update parameter with current portfolio value.

        :param current_portfolio_value: Current total value of the portfolio.
        :type current_portfolio_value: float
        :param kwargs: Unused arguments passed to :meth:`values_in_time`.
        :type kwargs: dict
        """
        self.rhs.value = self.c_min.current_value/current_portfolio_value

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return -w_plus[-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return -self.rhs


class LongCash(MinCashBalance):
    r"""Require that post-trade cash account is non-negative.

    In our notation, this is

    .. math::

        {(w_t + z_t)}_{n+1} \geq 0.

    Be mindful that trading costs (if present) are deducted from the cash
    account after trading, and so the next period's cash account value
    may be negative.
    """

    def __init__(self):
        super().__init__(0.)


class DollarNeutral(EqualityConstraint):
    r"""Long-short dollar neutral strategy.

    In our notation, this is

    .. math::

        \mathbf{1}^T \max({(w_t + z_t)}_{1:n}, 0) =
            -\mathbf{1}^T \min({(w_t + z_t)}_{1:n}, 0)

    which is simply :math:`{(w_t + z_t)}_{n+1} = 1`.
    """

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus[-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return 1


class MaxWeights(InequalityConstraint):
    r"""A max limit on post-trade weights (excluding cash).

    In our notation, this is

    .. math::

        {(w_t + z_t)}_{1:n} \leq w^\text{max}

    where the limit :math:`w^\text{max}` is either a scalar or a vector, see
    below.

    :param limit: A series or number giving the weights limit. See the
        :ref:`passing-data` manual page for details on how to provide this
        data. For example, you pass a float if you want a constant limit
        for all assets at all times, a Pandas series indexed by time if you
        want a limit constant for all assets but varying in time, a Pandas
        series indexed by the assets' names if you have limits constant in time
        but different for each asset, and a Pandas dataframe indexed by time
        and with assets as columns if you have a different limit for each point
        in time and each asset. If the value changes for each asset, you should
        provide a value for each name that ever appear in a back-test; the
        data will be sliced according to the current trading universe during a
        back-test. It is fine to have missing values at certain times on assets
        that are not traded then.
    :type limit: float, pandas.Series, pandas.DataFrame
    """

    def __init__(self, limit):
        self.limit = DataEstimator(limit, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.limit.parameter


class MinWeights(InequalityConstraint):
    r"""A min limit on post-trade weights (excluding cash).

    In our notation, this is

    .. math::

        {(w_t + z_t)}_{1:n} \geq w^\text{min}

    where the limit :math:`w^\text{min}` is either a scalar or a vector, see
    below.

    :param limit: A series or number giving the weights limit. See the
        :ref:`passing-data` manual page for details on how to provide this
        data. For example, you pass a float if you want a constant limit
        for all assets at all times, a Pandas series indexed by time if you
        want a limit constant for all assets but varying in time, a Pandas
        series indexed by the assets' names if you have limits constant in time
        but different for each asset, and a Pandas dataframe indexed by time
        and with assets as columns if you have a different limit for each point
        in time and each asset. If the value changes for each asset, you should
        provide a value for each name that ever appear in a back-test; the
        data will be sliced according to the current trading universe during a
        back-test. It is fine to have missing values at certain times on assets
        that are not traded then.
    :type limit: float, pandas.Series, pandas.DataFrame
    """

    def __init__(self, limit):
        self.limit = DataEstimator(limit, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return -w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return -self.limit.parameter

class MaxBenchmarkDeviation(MaxWeights):
    r"""A max limit on post-trade weights minus the benchmark weights.

    In our notation, this is

    .. math::

        {(w_t + z_t - w^\text{bm}_t)}_{1:n} \leq w^\text{max}

    where the limit :math:`w^\text{max}` is either a scalar or a vector, see
    below.

    .. versionadded:: 1.1.0
        Added in version 1.1.0

    :param limit: A series or number giving the weights limit. See the
        :ref:`passing-data` manual page for details on how to provide this
        data. For example, you pass a float if you want a constant limit
        for all assets at all times, a Pandas series indexed by time if you
        want a limit constant for all assets but varying in time, a Pandas
        series indexed by the assets' names if you have limits constant in time
        but different for each asset, and a Pandas dataframe indexed by time
        and with assets as columns if you have a different limit for each point
        in time and each asset. If the value changes for each asset, you should
        provide a value for each name that ever appear in a back-test; the
        data will be sliced according to the current trading universe during a
        back-test. It is fine to have missing values at certain times on assets
        that are not traded then.
    :type limit: float, pandas.Series, pandas.DataFrame
    """

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus_minus_w_bm[:-1]


class MinBenchmarkDeviation(MinWeights):
    r"""A min limit on post-trade weights minus the benchmark weights.

    In our notation, this is

    .. math::

        {(w_t + z_t - w^\text{bm}_t)}_{1:n} \geq w^\text{min}

    where the limit :math:`w^\text{min}` is either a scalar or a vector, see
    below.

    .. versionadded:: 1.1.0
        Added in version 1.1.0

    :param limit: A series or number giving the weights limit. See the
        :ref:`passing-data` manual page for details on how to provide this
        data. For example, you pass a float if you want a constant limit
        for all assets at all times, a Pandas series indexed by time if you
        want a limit constant for all assets but varying in time, a Pandas
        series indexed by the assets' names if you have limits constant in time
        but different for each asset, and a Pandas dataframe indexed by time
        and with assets as columns if you have a different limit for each point
        in time and each asset. If the value changes for each asset, you should
        provide a value for each name that ever appear in a back-test; the
        data will be sliced according to the current trading universe during a
        back-test. It is fine to have missing values at certain times on assets
        that are not traded then.
    :type limit: float, pandas.Series, pandas.DataFrame
    """

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return -w_plus_minus_w_bm[:-1]


class MinMaxWeightsAtTimes(Estimator):
    """This class abstracts functionalities used by the two below.

    .. note::
        This class is experimental and its interface may change, or we
        may drop it.

    :param base_limit: Limit of the constraint (either ``<=`` or ``>=``).
    :type base_limit: float
    :param times: Times at which the class is active.
    :type times: iterable of pandas.Timestamp
    """

    sign = 0 # should be 1 or -1

    def __init__(self, limit, times):
        self.base_limit = limit
        self.times = times
        self.limit = None
        self.trading_calendar = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, trading_calendar, **kwargs):
        """Initialize estimator instance with updated trading_calendar.

        :param trading_calendar: Future (including current) trading calendar.
        :type trading_calendar: pandas.DatetimeIndex
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.trading_calendar = trading_calendar
        self.limit = cp.Parameter()

    def values_in_time( # pylint: disable=arguments-differ
            self, t, mpo_step, **kwargs):
        """If target period is in sight activate constraint.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param mpo_step: Which step of a
            :class:`cvxportfolio.MultiPeriodOptimization` policy we're at.
        :type mpo_step: int
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        tidx = self.trading_calendar.get_loc(t)
        nowtidx = tidx + mpo_step
        if (nowtidx < len(self.trading_calendar)) and\
                (self.trading_calendar[nowtidx] in self.times):
            self.limit.value = self.base_limit
        else:
            self.limit.value = 100 * self.sign


class MinWeightsAtTimes(MinMaxWeightsAtTimes, InequalityConstraint):
    """Require that at certain times the weights are larger than a constant.

    .. note::
        This constraint is experimental and its interface may change, or we
        may drop it.

    :param base_limit: Minimum limit of the weights.
    :type base_limit: float
    :param times: Times at which the constraint is active.
    :type times: iterable of pandas.Timestamp
    """

    sign = -1.  # used in values_in_time of parent class

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return -w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return -self.limit


class MaxWeightsAtTimes(MinMaxWeightsAtTimes, InequalityConstraint):
    """Require that at certain times the weights are smaller than a constant.

    .. note::
        This constraint is experimental and its interface may change, or we
        may drop it.

    :param base_limit: Maximum limit of the weights.
    :type base_limit: float
    :param times: Times at which the constraint is active.
    :type times: iterable of pandas.Timestamp
    """
    sign = 1.  # used in values_in_time of parent class

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.limit


class FactorMaxLimit(InequalityConstraint):
    r"""A max limit on portfolio-wide factor (e.g. beta) exposure.

    It models the term:

    .. math::

        f_t^T {(w_t^+)}_{1:n} \leq l_t

    where :math:`f_t` is the factor exposure vector and
    :math:`l_t` the limit at time :math:`t`. It can also model a vector
    constraint

    .. math::

        F_t^T {(w_t^+)}_{1:n} \leq l_t

    where :math:`F_t` is a matrix of factor exposures and
    :math:`l_t` a vector of limits at time :math:`t`. See below for details.

    :param factor_exposure: Series or DataFrame giving the factor exposure.
        If Series it is indexed by assets' names and represents factor
        exposures constant in time. If DataFrame it is indexed by time
        and has the assets names as columns, and it represents factor
        exposures that change in time. In the latter case an observation
        must be present for every point in time of a backtest.
        If you want you can also pass multiple factor exposures at once:
        as a dataframe indexed by assets' names and whose columns are the
        factors (if constant in time), or a dataframe with multiindex:
        first level is time, second level are assets' names (if changing
        in time). However this latter usecase is probably better served
        by making multiple instances of this constraint, one for each
        factor.
    :type factor_exposure: pd.Series or pd.DataFrame
    :param limit: Factor limit, either constant or varying in time. Use
        a DataFrame if you pass multiple factors as once.
    :type limit: float or pd.Series or pd.DataFrame
    """

    def __init__(self, factor_exposure, limit):
        self.factor_exposure = DataEstimator(
            factor_exposure, compile_parameter=True)
        self.limit = DataEstimator(limit, compile_parameter=True,
            ignore_shape_check=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return self.factor_exposure.parameter.T @ w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.limit.parameter


class FactorMinLimit(InequalityConstraint):
    r"""A min limit on portfolio-wide factor (e.g. beta) exposure.

    It models the term:

    .. math::

        f_t^T {(w_t^+)}_{1:n} \geq l_t

    where :math:`f_t` is the factor exposure vector and
    :math:`l_t` the limit at time :math:`t`. It can also model a vector
    constraint

    .. math::

        F_t^T {(w_t^+)}_{1:n} \geq l_t

    where :math:`F_t` is a matrix of factor exposures and
    :math:`l_t` a vector of limits at time :math:`t`. See below for details.

    :param factor_exposure: Series or DataFrame giving the factor exposure.
        If Series it is indexed by assets' names and represents factor
        exposures constant in time. If DataFrame it is indexed by time
        and has the assets names as columns, and it represents factor
        exposures that change in time. In the latter case an observation
        must be present for every point in time of a backtest.
        If you want you can also pass multiple factor exposures at once:
        as a dataframe indexed by assets' names and whose columns are the
        factors (if constant in time), or a dataframe with multiindex:
        first level is time, second level are assets' names (if changing
        in time). However this latter usecase is probably better served
        by making multiple instances of this constraint, one for each
        factor.
    :type factor_exposure: pd.Series or pd.DataFrame
    :param limit: Factor limit, either constant or varying in time. Use
        a DataFrame if you pass multiple factors as once.
    :type limit: float or pd.Series or pd.DataFrame
    """

    def __init__(self, factor_exposure, limit):
        self.factor_exposure = DataEstimator(
            factor_exposure, compile_parameter=True)
        self.limit = DataEstimator(limit, compile_parameter=True,
            ignore_shape_check=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return -self.factor_exposure.parameter.T @ w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return -self.limit.parameter


class FactorGrossLimit(InequalityConstraint):
    r"""A gross limit on portfolio-wide factor (e.g. beta) exposure.

    It models the term:

    .. math::

        f_t^T |{(w_t^+)}_{1:n}| \leq l_t

    where :math:`f_t` is the factor exposure vector and
    :math:`l_t` the limit at time :math:`t`. It can also model a vector
    constraint

    .. math::

        F_t^T |{(w_t^+)}_{1:n}| \leq l_t

    where :math:`F_t` is a matrix of factor exposures and
    :math:`l_t` a vector of limits at time :math:`t`. See below for details.

    :param factor_exposure: Series or DataFrame giving the factor exposure.
        All elements must be non-negative.
        If Series it is indexed by assets' names and represents factor
        exposures constant in time. If DataFrame it is indexed by time
        and has the assets names as columns, and it represents factor
        exposures that change in time. In the latter case an observation
        must be present for every point in time of a backtest.
        If you want you can also pass multiple factor exposures at once:
        as a dataframe indexed by assets' names and whose columns are the
        factors (if constant in time), or a dataframe with multiindex:
        first level is time, second level are assets' names (if changing
        in time). However this latter usecase is probably better served
        by making multiple instances of this constraint, one for each
        factor.
    :type factor_exposure: pd.Series or pd.DataFrame
    :param limit: Factor limit, either constant or varying in time. Use
        a DataFrame if you pass multiple factors as once.
    :type limit: float or pd.Series or pd.DataFrame
    """

    def __init__(self, factor_exposure, limit):
        self.factor_exposure = DataEstimator(
            factor_exposure, non_negative=True, compile_parameter=True)
        self.limit = DataEstimator(limit, compile_parameter=True,
            ignore_shape_check=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return self.factor_exposure.parameter.T @ cp.abs(w_plus[:-1])

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.limit.parameter


class FixedFactorLoading(EqualityConstraint):
    r"""A constraint to fix portfolio loadings to a set of factors.

    This can be used to impose market neutrality,
    a certain portfolio-wide alpha, ....

    It models the term:

    .. math::

        f_t^T {(w_t^+)}_{1:n} = l_t

    where :math:`f_t` is the factor exposure vector and
    :math:`l_t` the limit at time :math:`t`. It can also model a vector
    constraint

    .. math::

        F_t^T {(w_t^+)}_{1:n} = l_t

    where :math:`F_t` is a matrix of factor exposures and
    :math:`l_t` a vector of limits at time :math:`t`. See below for details.

    :param factor_exposure: Series or DataFrame giving the factor exposure.
        If Series it is indexed by assets' names and represents factor
        exposures constant in time. If DataFrame it is indexed by time
        and has the assets names as columns, and it represents factor
        exposures that change in time. In the latter case an observation
        must be present for every point in time of a backtest.
        If you want you can also pass multiple factor exposures at once:
        as a dataframe indexed by assets' names and whose columns are the
        factors (if constant in time), or a dataframe with multiindex:
        first level is time, second level are assets' names (if changing
        in time). However this latter usecase is probably better served
        by making multiple instances of this constraint, one for each
        factor.
    :type factor_exposure: pd.Series or pd.DataFrame
    :param target: Target portfolio factor exposures,
        either constant or varying in time. Use
        a DataFrame if you pass multiple factors as once.
    :type target: float or pd.Series or pd.DataFrame
    """

    def __init__(self, factor_exposure, target):
        self.factor_exposure = DataEstimator(
            factor_exposure, compile_parameter=True)
        self.target = DataEstimator(target, compile_parameter=True)

    def _compile_constr_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile left hand side of the constraint expression."""
        return self.factor_exposure.parameter.T @ w_plus[:-1]

    def _rhs(self):
        """Compile right hand side of the constraint expression."""
        return self.target.parameter

class FactorNeutral(FixedFactorLoading):
    r"""Require neutrality with respect to certain risk factors.

    This is developed at :paper:`page 35 of the paper <section.4.4>`.
    This models the term

    .. math::

        {(f_t)}^T {(w_t^+)}_{1:n} = 0

    where :math:`f_t` is the factor exposure vector. It can also model
    a vector constraint

    .. math::

        F_t^T {(w_t^+)}_{1:n} = 0

    where :math:`F_t` is a matrix of factor exposures. See below for details.

    :param factor_exposure: Series or DataFrame giving the factor exposure.
        If Series it is indexed by assets' names and represents factor
        exposures constant in time. If DataFrame it is indexed by time
        and has the assets names as columns, and it represents factor
        exposures that change in time. In the latter case an observation
        must be present for every point in time of a backtest.
        If you want you can also pass multiple factor exposures at once:
        as a dataframe indexed by assets' names and whose columns are the
        factors (if constant in time), or a dataframe with multiindex:
        first level is time, second level are assets' names (if changing
        in time). However this latter usecase is probably better served
        by making multiple instances of this constraint, one for each
        factor.
    :type factor_exposure: pd.Series or pd.DataFrame
    """

    def __init__(self, factor_exposure):
        super().__init__(factor_exposure=factor_exposure, target=0.)
