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
r"""This module implements the simple forecasting models used by Cvxportfolio.

These are standard ones like historical mean, variance, and covariance. In
most cases the models implemented here are equivalent to the relevant Pandas
DataFrame methods, including (most importantly) the logic used to skip over any
``np.nan``. There are some subtle differences explained below.

Our forecasters are optimized to be evaluated sequentially in time: at each
point in time in a back-test the forecast computed at the previous time step
is updated with the most recent observation. This is in some cases (e.g.,
covariances) much more efficient than computing from scratch.

Most of our forecasters implement both a rolling window and exponential moving
average logic. These are specified by the ``rolling`` and ``half_life``
parameters respectively, which are either Pandas Timedeltas or ``np.inf``.
The latter is the default, and means that the whole past is used, with no
exponential smoothing. Note that it's possible to use both, *e.g.*,
estimate covariance matrices ignoring past returns older than 5 years and
smoothing the recent ones using an exponential kernel with half-life of 1 year.

Finally, we note that the covariance, variance and standard deviation
forecasters implement the ``kelly`` parameter, which is True by default.
This is a simple trick explained in
:paper:`section 4.2 (page 28) <section.4.2>` of the paper, simplifies the
computation and provides in general (slightly) higher performance.
For example, using the notation of the paper, the classical definition of
covariance is

.. math::

    \Sigma = \mathbf{E}(r_t - \mu)(r_t - \mu)^T,

this is what you get by setting ``kelly=False``. The default, ``kelly=True``,
gives instead

.. math::

    \Sigma^\text{kelly} = \mathbf{E}r_t r_t^T = \Sigma + \mu \mu^T,

so that the resulting Markowitz-style optimization problem corresponds to
the second order Taylor approximation of a (risk-constrained) Kelly objective,
as is explained briefly :paper:`at page 28 of the paper <section.4.2>`, or with
more detail (and hard-to-read math) in `section 6 of the Risk-Constrained Kelly
Gambling paper
<https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf#section.6>`_.

Lastly, some forecasters implement a basic caching mechanism.
This is used in two ways. First, online (e.g., in back-test): if multiple
copies of the same forecaster need access to the estimated value, as is the
case in :class:`cvxportfolio.MultiPeriodOptimization` policies, the expensive
evaluation is only done once. Then, offline, provided that the
:class:`cvxportfolio.data.MarketData` server used implements the
:meth:`cvxportfolio.data.MarketData.partial_universe_signature` method
(so that we can certify which market data the cached values are computed on).
This type of caching simply saves on disk the forecasted values, and makes it
available automatically next time the user runs a back-test on the same market
data (and same universe). This is especially useful when doing hyper-parameter
optimization, so that expensive computations like evaluating large covariance
matrices are only done once.

How to use them
~~~~~~~~~~~~~~~

These forecasters are each the default option of some Cvxportfolio optimization
term, for example :class:`HistoricalMeanReturn` is the default used by
:class:`cvxportfolio.ReturnsForecast`. In this way each is used with its
default options. If you want to change the options you can simply pass
the relevant forecaster class, instantiated with the options of your choice,
to the Cvxportfolio object. For example

.. code-block::

    import cvxportfolio as cvx
    from cvxportfolio.forecast import HistoricalMeanReturn
    import pandas as pd

    returns_forecast = cvx.ReturnsForecast(
        r_hat = HistoricalMeanReturn(
            half_life=pd.Timedelta(days=365),
            rolling=pd.Timedelta(days=365*5)))

if you want to apply exponential smoothing to the mean returns forecaster with
half-life of 1 year, and skip over all observations older than 5 years. Both
are relative to each point in time at which the policy is evaluated.
"""

import logging

import numpy as np
import pandas as pd

from .errors import DataError, ForecastError
from .estimator import DataEstimator, Estimator, SimulatorEstimator
from .hyperparameters import _resolve_hyperpar

logger = logging.getLogger(__name__)


class BaseForecast(Estimator):
    """Base class for forecasters."""

    _last_time = None

    # Will be exposed to the user, for now it's a class-level constant
    _CACHED = False

    def values_in_time_recursive(self, t, cache=None, **kwargs):
        """Override default method to handle caching.

        :param t: Current timestamp in execution or back-test.
        :type t: pd.Timestamp
        :param cache: Cache dictionary, if available. Default None.
        :type cache: dict or None
        :param kwargs: Various arguments to :meth:`values_in_time_recursive`.
        :type kwargs: dict
        """

        if not self._CACHED:
            return super().values_in_time_recursive(t=t, cache=cache, **kwargs)
        else:
            # this part is copied from Estimator
            # TODO: could refactor upstream to avoid copy-pasting these clauses
            for _, subestimator in self.__dict__.items():
                if hasattr(subestimator, "values_in_time_recursive"):
                    subestimator.values_in_time_recursive(
                        t=t, cache=cache, **kwargs)
            for subestimator in self.__subestimators__:
                subestimator.values_in_time_recursive(
                    t=t, cache=cache, **kwargs)

        # here goes caching
        if hasattr(self, "values_in_time"):

            if cache is None:  # e.g., in execute() cache is disabled
                cache = {}

            if str(self) not in cache:
                cache[str(self)] = {}

            if t in cache[str(self)]:
                logger.info(
                    '%s.values_in_time at time %s is retrieved from cache.',
                    self, t)
                self._current_value = cache[str(self)][t]
            else:
                self._current_value = self.values_in_time(
                    t=t, cache=cache, **kwargs)
                logger.info('%s.values_in_time at time %s is stored in cache.',
                    self, t)
                cache[str(self)][t] = self._current_value
            return self.current_value

        return None # pragma: no cover

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self._last_time = None

    def estimate(self, market_data, t=None):
        """Estimate the forecaster at given time on given market data.

        This uses the same logic used by a trading policy to evaluate the
        forecaster at a given point in time.

        :param market_data: Market data server, used to provide data to the
            forecaster.
        :type market_data: cvx.MarketData instance
        :param t: Time at which to estimate the forecaster. Must be among
            the ones returned by ``market_data.trading_calendar()``. Default is
            ``None``, meaning that the last valid timestamp is chosen. Note
            that with default market data servers you need to set
            ``online_usage=True`` if forecasting on the last timestamp
            (usually, today).
        :type t: pd.Timestamp or None

        .. note::

            This method is not finalized! It is still experimental, and not
            covered by semantic versioning guarantees.

        :raises ValueError: If the provided time t is not in the trading
            calendar.

        :returns: Forecasted value and time at which the forecast is made
            (for safety checking).
        :rtype: (np.array, pd.Timestamp)
        """

        trading_calendar = market_data.trading_calendar()

        if t is None:
            t = trading_calendar[-1]

        if not t in trading_calendar:
            raise ValueError(f'Provided time {t} must be in the '
            + 'trading calendar implied by the market data server.')

        past_returns, _, past_volumes, _, current_prices = market_data.serve(t)

        self.initialize_estimator_recursive(
            universe=past_returns.columns,
            trading_calendar=trading_calendar[trading_calendar >= t])

        forecast = self.values_in_time_recursive(
            t=t, past_returns=past_returns, past_volumes=past_volumes,
            current_weights=None, current_portfolio_value=None,
            current_prices=current_prices)

        self.finalize_estimator_recursive()

        return forecast, t

    def _agnostic_update(self, t, past_returns, **kwargs):
        """Choose whether to make forecast from scratch or update last one."""
        if (self._last_time is None) or (
            self._last_time != past_returns.index[-1]):
            logger.debug(
                '%s.values_in_time at time %s is computed from scratch.',
                self, t)
            self._initial_compute(t=t, past_returns=past_returns, **kwargs)
        else:
            logger.debug(
              '%s.values_in_time at time %s is updated from previous value.',
              self, t)
            self._online_update(t=t, past_returns=past_returns, **kwargs)

    def _initial_compute(self, **kwargs):
        """Make forecast from scratch."""
        raise NotImplementedError # pragma: no cover

    def _online_update(self, **kwargs):
        """Update forecast from period before."""
        raise NotImplementedError # pragma: no cover

def _is_timedelta(value):
    if isinstance(value, pd.Timedelta):
        if value <= pd.Timedelta('0d'):
            raise ValueError(
                '(Exponential) moving average window must be positive')
        return True
    if isinstance(value, float) and np.isposinf(value):
        return False
    raise ValueError(
        '(Exponential) moving average window can only be'
        ' pandas Timedeltas or np.inf.')


class BaseMeanVarForecast(BaseForecast):
    """This class contains logic common to mean and (co)variance forecasters.

    It implements both moving average and exponential moving average, which
    can be used at the same time (e.g., ignore observations older than 5
    years and weight exponentially with half-life of 1 year the recent ones).

    Then, it implements the "online update" vs "compute from scratch" model,
    updating with a new observations is much cheaper than computing from
    scratch (especially for covariances).
    """

    _denominator = None
    _numerator = None

    def __init__(self, half_life=np.inf, rolling=np.inf):
        self.half_life = half_life
        self.rolling = rolling

    def initialize_estimator( # pylint: disable=arguments-differ
            self, **kwargs):
        """Re-initialize whenever universe changes.

        :param kwargs: Unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        super().initialize_estimator(**kwargs)
        self._denominator = None
        self._numerator = None

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        raise NotImplementedError # pragma: no cover

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        raise NotImplementedError # pragma: no cover

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied in this class.
        """
        raise NotImplementedError # pragma: no cover

    def _dataframe_selector(self, **kwargs):
        """Return dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        raise NotImplementedError # pragma: no cover

    def _get_last_row(self, **kwargs):
        """Return last row of the dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.

        You may redefine it if obtaining the full dataframe is expensive,
        during online update (in most cases) only this method is required.
        """
        return self._dataframe_selector(**kwargs).iloc[-1]

    def values_in_time( # pylint: disable=arguments-differ
            self, **kwargs):
        """Obtain current value of the historical mean of given dataframe.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Historical means of given dataframe.
        :rtype: numpy.array
        """
        self._agnostic_update(**kwargs)
        return (self._numerator / self._denominator).values

    def _emw_weights(self, index, t):
        """Get weights to apply to the past obs for EMW."""
        index_in_halflifes = (index - t) / _resolve_hyperpar(self.half_life)
        return np.exp(index_in_halflifes * np.log(2))

    def _initial_compute(self, t, **kwargs): # pylint: disable=arguments-differ
        """Make forecast from scratch.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        df = self._dataframe_selector(t=t, **kwargs)

        # Moving average window logic
        if _is_timedelta(_resolve_hyperpar(self.rolling)):
            df = df.loc[df.index >= t-_resolve_hyperpar(self.rolling)]

        # If EMW, compute weights here
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(df.index, t)
        else:
            emw_weights = None

        self._denominator = self._compute_denominator(df, emw_weights)
        self._check_denominator_valid(t)
        self._numerator = self._compute_numerator(df, emw_weights)
        self._last_time = t

        # used by covariance forecaster
        return df, emw_weights

    def _online_update(self, t, **kwargs): # pylint: disable=arguments-differ
        """Update forecast from period before.

        This method receives the **kwargs passed to :meth:`values_in_time`.
        """
        last_row = self._get_last_row(t=t, **kwargs)

        # if emw discount past
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            time_passed_in_halflifes = (
                self._last_time - t)/_resolve_hyperpar(self.half_life)
            discount_factor = np.exp(time_passed_in_halflifes * np.log(2))
            self._denominator *= discount_factor
            self._numerator *= discount_factor
        else:
            discount_factor = 1.

        # for emw we also need to discount last element
        self._denominator += self._update_denominator(
            last_row) * discount_factor
        self._numerator += self._update_numerator(last_row) * discount_factor

        # Moving average window logic: subtract elements that have gone out
        if _is_timedelta(_resolve_hyperpar(self.rolling)):
            df = self._dataframe_selector(t=t, **kwargs)
            observations_to_subtract, emw_weights_of_subtract = \
                self._remove_part_gone_out_of_ma(df, t)
        else:
            observations_to_subtract, emw_weights_of_subtract = None, None

        self._last_time = t

        # used by covariance forecaster
        return (
            discount_factor, observations_to_subtract, emw_weights_of_subtract)

    def _remove_part_gone_out_of_ma(self, df, t):
        """Subtract from numerator and denominator too old observations."""

        observations_to_subtract = df.loc[
            (df.index >= (self._last_time - _resolve_hyperpar(self.rolling)))
            & (df.index < (t - _resolve_hyperpar(self.rolling)))]

        # If EMW, compute weights here
        if _is_timedelta(_resolve_hyperpar(self.half_life)):
            emw_weights = self._emw_weights(observations_to_subtract.index, t)
        else:
            emw_weights = None

        self._denominator -= self._compute_denominator(
            observations_to_subtract, emw_weights)
        self._check_denominator_valid(t)
        self._numerator -= self._compute_numerator(
            observations_to_subtract, emw_weights).fillna(0.)

        # used by covariance forecaster
        return observations_to_subtract, emw_weights

    def _check_denominator_valid(self, t):
        """Check that there are enough obs to compute the forecast."""
        mindenom = np.min(self._denominator.values)
        if mindenom == 0:
            raise ForecastError(
                f'{self.__class__.__name__} can not compute the forecast at'
                + f' time {t} because there are no observation for either some'
                ' asset or some pair of assets (in the case of covariance).')
        if mindenom < 5:
            logger.warning(
                '%s at time %s is given 5 or less observations for either some'
                + ' asset or some pair of assets (in the case of covariance).',
                self.__class__.__name__, t)


class BaseMeanForecast(BaseMeanVarForecast): # pylint: disable=abstract-method
    """This class contains the logic common to the mean forecasters."""

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        if emw_weights is None:
            return df.sum()
        return df.multiply(emw_weights, axis=0).sum()

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        if emw_weights is None:
            return df.count()
        ones = (~df.isnull()) * 1.
        return ones.multiply(emw_weights, axis=0).sum()

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        return last_row.fillna(0.)

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        return ~(last_row.isnull())


class HistoricalMeanReturn(BaseMeanForecast):
    r"""Historical means of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].mean()

    where ``past_returns`` is a time-indexed dataframe containing the past
    returns (if in back-test that's relative to each point in time, ), and its
    last column, which we skip over, are the cash returns. We use the same
    logic as Pandas to handle ``np.nan`` values.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical means of."""
        return past_returns.iloc[:, :-1]


class RegressionXtY(HistoricalMeanReturn):
    """Class for the XtY matrix of returns regression forecaster."""

    def __init__(self, regressor, **kwargs):
        assert isinstance(regressor, UserProvidedRegressor)
        super().__init__(**kwargs)
        self.regressor = regressor

    def _work_with(self, past_returns, **kwargs):
        """Base DataFrame we work with."""
        return past_returns.iloc[:, :-1]

    # pylint: disable=arguments-differ
    def _dataframe_selector(self, **kwargs):
        """Return dataframe to compute the historical means of."""
        regr_on_df = self.regressor._get_all_history(
            self._work_with(**kwargs).index)
        return self._work_with(
            **kwargs).multiply(regr_on_df, axis=0).dropna(how='all')

    def _get_last_row(self, **kwargs):
        """Return last row of the dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.

        You may redefine it if obtaining the full dataframe is expensive,
        during online update (in most cases) only this method is required.
        """
        return self._work_with(
            **kwargs).iloc[-1] * self.regressor.current_value


class UserProvidedRegressor(DataEstimator):
    """User provided regressor series."""

    def __init__(self, data, min_obs=10):
        assert isinstance(data, pd.Series)
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.name is not None
        assert isinstance(data.name, str)
        super().__init__(data, use_last_available_time=True)
        self._min_obs = min_obs

    def _get_all_history(self, pandas_obj_idx):
        """Get history of this regressor indexed on pandas obj."""
        result = self.data.reindex(
            pandas_obj_idx, method='ffill').dropna()
        if len(result) < self._min_obs:
            raise DataError(
                f'Regressor {self.name} at time {pandas_obj_idx[-1]} '
                f' has less history than min_obs={self._min_obs},'
                ' changing regressor in time is not (currently) supported.')
        return result

    @property
    def name(self):
        """Name of the regressor.

        :returns: Name
        :rtype: str
        """
        return self.data.name

class RegressionMeanReturn(BaseForecast):
    """Test class."""

    def __init__(self, regressors, **kwargs):
        # super().__init__(**kwargs)
        self.regressors = [
            UserProvidedRegressor(regressor) for regressor in regressors]

        self.XtY_forecasters = {
            regressor.name: RegressionXtY(regressor)
                for regressor in self.regressors}

        self.__subestimators__ = tuple(
            [HistoricalMeanReturn()] + self.regressors
            + list(self.XtY_forecasters.values()))

        self.XtX_matrices = None

    def initialize_estimator(self, universe, **kwargs):
        """Initialize, create XtX matrices knowing the current universe.

        :param universe: Current trading universe.
        :type universe: pd.Index
        :param **kwargs: Other arguments to :meth:`initialize_estimator`.
        :type **kwargs: dict
        """

        self.XtX_matrices = {
            asset: RegressorsXtXMatrix(
                col_name=asset, regressors=self.regressors)
                    for asset in universe[:-1]}

        for XtX in self.XtX_matrices.values():
            XtX.initialize_estimator_recursive(universe=universe, **kwargs)

        # this method is called *after* having iterated on __subestimators__,
        # so it's safe to do this
        self.__subestimators__ = tuple(
            list(self.__subestimators__ ) + list(self.XtX_matrices.values()))

    def finalize_estimator(self, **kwargs):
        """Remove XtX matrices at change of universe.

        :param **kwargs: Unused arguments to :meth:`finalize_estimator`.
        :type **kwargs: dict
        """

        self.__subestimators__ = tuple(
            [HistoricalMeanReturn()] + self.regressors
            + list(self.XtY_forecasters.values()))

        self.XtX_matrices = None

    def values_in_time(self, t, past_returns, **kwargs):
        """Do it from scratch."""
        assets = past_returns.columns[:-1]

        # print('all_X_matrices')
        # print(self.all_X_matrices)

        # self._all_XtY_means = {
        #     regressor.name: self.multiply_df_by_regressor(
        #         past_returns.iloc[:, :-1], regressor).mean()
        #             for regressor in self.regressors}

        test = {
            regressor.name: pd.Series(self.XtY_forecasters[regressor.name].current_value, past_returns.columns[:-1])
                    for regressor in self.regressors}

        # print(test)
        # print(self._all_XtY_means)
        # assert np.allclose(pd.Series(test['VIX']), pd.Series(self._all_XtY_means['VIX']))
        self._all_XtY_means = test
        # raise Exception

        self._all_XtY_means[
            'intercept'] = pd.Series(
                self.__subestimators__[0].current_value,
                past_returns.columns[:-1])

        # print('all_XtY_means')
        # print(self.all_XtY_means)

        X_last = pd.Series(1., index=['intercept'])
        for regressor in self.regressors:
            X_last[regressor.name] = regressor.current_value

        # print('X_last')
        # print(X_last)

        all_solves = {
            asset: self.solve_for_single_X(
                asset, X_last, quad_reg=0.) for asset in assets}

        # print('all_solves')
        # print(all_solves)

        # result should be an array
        result = pd.Series(index = assets, dtype=float)
        for asset in assets:
            result[asset] = np.dot(
                all_solves[asset],
                    [self._all_XtY_means[regressor][asset]
                        for regressor in all_solves[asset].index])

        return result

    def solve_for_single_X(self, asset, X_last, quad_reg):
        """Solve with X_last."""
        XtX_mean = self.XtX_matrices[asset].current_value

        tikho_diag = np.array(np.diag(XtX_mean))
        tikho_diag[0] = 0. # intercept
        return pd.Series(np.linalg.solve(
            XtX_mean + np.diag(tikho_diag * quad_reg), X_last), X_last.index)

    # @staticmethod
    # def multiply_df_by_regressor(df, regressor):
    #     """Multiply time-indexed dataframe by time-indexed regressor.

    #     At each point in time, use last available observation of the regressor.
    #     """
    #     regr_on_df = regressor._get_all_history(df)
    #     return df.multiply(regr_on_df, axis=0).dropna(how='all')


class HistoricalMeanVolume(BaseMeanForecast):
    r"""Historical means of traded volume in units of value (e.g., dollars).

    .. versionadded:: 1.2.0

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    """
    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_volumes, **kwargs):
        """Return dataframe to compute the historical means of."""
        if past_volumes is None:
            raise DataError(
                f"{self.__class__.__name__} can only be used if MarketData"
                + " provides market volumes.")
        return past_volumes

class HistoricalVariance(BaseMeanForecast):
    r"""Historical variances of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].var(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        (past_returns**2).iloc[:,:-1].mean()

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` compute :math:`\mathbf{E}[r^2]`, else
        :math:`\mathbf{E}[r^2] - {\mathbf{E}[r]}^2`. The second corresponds
        to the classic definition of variance, while the first is what is
        obtained by Taylor approximation of the Kelly gambling objective.
        See discussion above.
    :type kelly: bool
    """

    def __init__(self, rolling=np.inf, half_life=np.inf, kelly=True):
        super().__init__(rolling=rolling, half_life=half_life)
        self.kelly = kelly

        if not self.kelly:
            self.meanforecaster = HistoricalMeanReturn(
                half_life=_resolve_hyperpar(self.half_life),
                rolling=_resolve_hyperpar(self.rolling))

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Variances of past returns (excluding cash).
        :rtype: numpy.array
        """
        result = super().values_in_time(**kwargs)
        if not self.kelly:
            result -= self.meanforecaster.current_value**2
        return result

    # pylint: disable=arguments-differ
    def _dataframe_selector(self, past_returns, **kwargs):
        """Return dataframe to compute the historical means of."""
        return past_returns.iloc[:, :-1]**2


class HistoricalStandardDeviation(HistoricalVariance, SimulatorEstimator):
    """Historical standard deviation of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent to

    .. code-block::

        past_returns.iloc[:,:-1].std(ddof=0)

    if you set ``kelly=False`` and

    .. code-block::

        np.sqrt((past_returns**2).iloc[:,:-1].mean())

    otherwise (we use the same logic to handle ``np.nan`` values).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default True.
    :type kelly: bool
    """

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviations of past returns (excluding cash).
        :rtype: numpy.array
        """
        variances = \
            super().values_in_time(**kwargs)
        return np.sqrt(variances)

    def simulate(self, **kwargs):
        # TODO could take last return as well
        return self.values_in_time(
            t=kwargs['t'],
            # These are not necessary with current design of
            # DataEstimator
            current_weights=kwargs['current_weights'],
            current_portfolio_value=kwargs['current_portfolio_value'],
            past_returns=kwargs['past_returns'],
            past_volumes=kwargs['past_volumes'],
            current_prices=kwargs['current_prices']
        )

class HistoricalMeanError(HistoricalVariance):
    r"""Historical standard deviations of the mean of non-cash returns.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    For a given time series of past returns :math:`r_{t-1}, r_{t-2},
    \ldots, r_0` this is :math:`\sqrt{\text{Var}[r]/t}`. When there are
    missing values we ignore them, both to compute the variance and the
    count.

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. If in back-test, that is with respect to each point
        in time. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: Same as in :class:`cvxportfolio.forecast.HistoricalVariance`.
        Default False.
    :type kelly: bool
    """

    def __init__(self, rolling=np.inf, half_life=np.inf, kelly=False):
        super().__init__(rolling=rolling, half_life=half_life, kelly=kelly)

    def values_in_time(self, **kwargs):
        """Obtain current value either by update or from scratch.

        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Standard deviation of the mean of past returns (excluding
            cash).
        :rtype: numpy.array
        """
        variance = super().values_in_time(**kwargs)
        return np.sqrt(variance / self._denominator.values)


class HistoricalCovariance(BaseMeanVarForecast):
    r"""Historical covariance matrix."""

    _joint_mean = None

    def __init__(self, rolling=np.inf, half_life=np.inf, kelly=True):
        super().__init__(rolling=rolling, half_life=half_life)
        self.kelly = kelly

    def initialize_estimator(self, **kwargs):
        super().initialize_estimator(**kwargs)
        self._joint_mean = None

    def _compute_numerator(self, df, emw_weights):
        """Exponential moving window (optional) numerator."""
        filled = df.fillna(0.)
        if emw_weights is None:
            return filled.T @ filled
        tmp = filled.multiply(emw_weights, axis=0)
        return tmp.T @ filled

    def _compute_denominator(self, df, emw_weights):
        """Exponential moving window (optional) denominator."""
        ones = (~df.isnull()) * 1.
        if emw_weights is None:
            return ones.T @ ones
        tmp = ones.multiply(emw_weights, axis=0)
        return tmp.T @ ones

    def _update_denominator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        nonnull = ~(last_row.isnull())
        return np.outer(nonnull, nonnull)

    def _update_numerator(self, last_row):
        """Update with last observation.

        Emw (if any) is applied upstream.
        """
        filled = last_row.fillna(0.)
        return np.outer(filled, filled)

    def _dataframe_selector( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Return dataframe to compute the historical covariance of."""
        return past_returns.iloc[:, :-1]

    def _compute_joint_mean(self, df, emw_weights):
        r"""Compute precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        nonnull = (~df.isnull()) * 1.
        if emw_weights is None:
            return nonnull.T @ df.fillna(0.)
        return nonnull.T @ df.fillna(0.).multiply(emw_weights, axis=0)

    def _update_joint_mean(self, last_row):
        r"""Update precursor of :math:`\Sigma_{i,j} =
        \mathbf{E}[r^{i}]\mathbf{E}[r^{j}]`."""
        return last_row.fillna(0.)

    def _initial_compute( # pylint: disable=arguments-differ
            self, **kwargs):
        """Compute from scratch, taking care of non-Kelly correction."""

        df, emw_weights = super()._initial_compute(**kwargs)

        if not self.kelly:
            self._joint_mean = self._compute_joint_mean(df, emw_weights)

    def _online_update( # pylint: disable=arguments-differ
            self, **kwargs):
        """Update from last observation."""

        discount_factor, observations_to_subtract, emw_weights_of_subtract = \
            super()._online_update(**kwargs)
        last_row = self._get_last_row(**kwargs)

        if not self.kelly:

            # discount past if EMW
            if discount_factor != 1.:
                self._joint_mean *= discount_factor

            # add last anyways
            self._joint_mean += self._update_joint_mean(
                last_row) * discount_factor

            # if MW, update by removing old observations
            if observations_to_subtract is not None:
                self._joint_mean -= self._compute_joint_mean(
                    observations_to_subtract, emw_weights_of_subtract)

    def values_in_time(self, **kwargs):
        """Obtain current value of the covariance estimate.

        :param kwargs: All arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        covariance = super().values_in_time(**kwargs)

        if not self.kelly:
            tmp = self._joint_mean / self._denominator
            covariance -= tmp.T * tmp

        return covariance

class RegressorsXtXMatrix(HistoricalCovariance):
    """XtX matrix used for linear regression.

    The user doesn't interact with this class directly, it is managed by the
    regression forecasters.
    """

    def __init__(self, regressors, col_name, **kwargs):
        self.regressors = regressors
        self.col_name = col_name
        super().__init__(kelly=True, **kwargs)

    def _work_with(self, past_returns, **kwargs):
        """Which base dataframe do we work with.

        This receives the arguments to :meth:`values_in_time`.

        In this class we only use its missing values structure, not its values!

        :param past_returns: Past market returns.
        :type past_returns: pd.DataFrame
        """
        return past_returns

    def _dataframe_selector(self, **kwargs):
        """Return dataframe we work with."""
        # TODO: might take just the index here
        col_of_df = self._work_with(**kwargs)[self.col_name].dropna()
        regr_on_col = []
        for regressor in self.regressors:
            regr_on_col.append(regressor._get_all_history(col_of_df.index))
        ones = pd.Series(1., col_of_df.index, name='intercept')
        res = pd.concat([ones] + regr_on_col, axis=1, sort=True)
        # print(res)
        return res

    def _get_last_row(self, **kwargs):
        """Return last row of the dataframe we work with.

        This method receives the **kwargs passed to :meth:`values_in_time`.

        During online update (in most cases) only this method is required,
        so :meth:`_dataframe_selector` is not called.
        """
        #result = self._dataframe_selector(**kwargs).iloc[-1]
        result1 = pd.Series(1., ['intercept'])
        for regressor in self.regressors:
            result1[regressor.name] = regressor.current_value
        #assert np.all(result == result1)
        # raise Exception
        return result1


def project_on_psd_cone_and_factorize(covariance):
    """Factorize matrix and remove negative eigenvalues.

    :param covariance: Square (symmetric) approximate covariance matrix, can
         have negative eigenvalues.
    :type covariance: numpy.array

    :returns: Square root factorization with negative eigenvalues removed.
    :rtype: numpy.array
    """
    eigval, eigvec = np.linalg.eigh(covariance)
    eigval = np.maximum(eigval, 0.)
    return eigvec @ np.diag(np.sqrt(eigval))

class HistoricalFactorizedCovariance(HistoricalCovariance):
    r"""Historical covariance matrix of non-cash returns, factorized.

    .. versionadded:: 1.2.0

        Added the ``half_life`` and ``rolling`` parameters.

    When both ``half_life`` and ``rolling`` are infinity, this is equivalent
    to, before factorization

    .. code-block::

        past_returns.iloc[:,:-1].cov(ddof=0)

    if you set ``kelly=False``. We use the same logic to handle ``np.nan``
    values. For ``kelly=True`` it is not possible to reproduce with one single
    Pandas method (but we do test against Pandas in the unit tests).

    :param half_life: Half-life of exponential smoothing, expressed as
        Pandas Timedelta. Default ``np.inf``, meaning no exponential smoothing.
    :type half_life: pandas.Timedelta or np.inf
    :param rolling: Rolling window used: observations older than this Pandas
        Timedelta are skipped over. If in back-test, that is with respect to
        each point in time. Default ``np.inf``, meaning that all past is used.
    :type rolling: pandas.Timedelta or np.inf
    :param kelly: if ``True`` each element of the covariance matrix
        :math:`\Sigma_{i,j}` is equal to :math:`\mathbf{E} r^{i} r^{j}`,
        otherwise it is
        :math:`\mathbf{E} r^{i} r^{j} - \mathbf{E} r^{i} \mathbf{E} r^{j}`.
        The second case corresponds to the classic definition of covariance,
        while the first is what is obtained by Taylor approximation of
        the Kelly gambling objective. (See discussion above.)
        In the second case, the estimated covariance is the same
        as what is returned by ``pandas.DataFrame.cov(ddof=0)``, *i.e.*,
        we use the same logic to handle missing data.
    :type kelly: bool
    """

    # this is used by FullCovariance
    FACTORIZED = True

    # Will be exposed to the user, for now it's a class level constant
    _CACHED = True

    def values_in_time( # pylint: disable=arguments-differ
            self, t, **kwargs):
        """Obtain current value of the covariance estimate.

        :param t: Current time period (possibly of simulation).
        :type t: pandas.Timestamp
        :param kwargs: All arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.ForecastError: The procedure failed,
            typically because there are too many missing values (*e.g.*, some
            asset has only missing values).

        :returns: Square root factorized covariance matrix (excludes cash).
        :rtype: numpy.array
        """

        covariance = super().values_in_time(t=t, **kwargs)

        try:
            return project_on_psd_cone_and_factorize(covariance)
        except np.linalg.LinAlgError as exc: # pragma: no cover
            raise ForecastError(f'Covariance estimation at time {t} failed;'
                + ' there are (probably) too many missing values in the'
                + ' past returns.') from exc


class HistoricalLowRankCovarianceSVD(BaseForecast):
    """Build factor model covariance using truncated SVD.

    .. note::

        This forecaster is experimental and not covered by semantic versioning,
        we may change it without warning.

    :param num_factors: How many factors in the low rank model.
    :type num_factors: int
    :param svd_iters: How many iteration of truncated SVD to apply. If you
        get a badly conditioned covariance you may to lower this.
    :type svd_iters: int
    :param svd: Which SVD routine to use, currently only dense (LAPACK) via
        Numpy.
    :type svd: str
    """

    def __init__(self, num_factors, svd_iters=10, svd='numpy'):
        self.num_factors = num_factors
        self.svd_iters = svd_iters
        self.svd = svd

    # num_factors: int
    # svd_iters: int = 10
    # svd: str = 'numpy'

    # brought back from old commit;
    #
    #  https://github.com/cvxgrp/cvxportfolio/commit
    #     /aa3d2150d12d85a6fb1befdf22cb7967fcc27f30
    #
    # matches original 2016 method from example
    # notebooks with new heuristic for NaNs. can probably be improved
    # by terminating early if idyosyncratic becomes negative

    @staticmethod
    def build_low_rank_model(rets, num_factors=10, iters=10, svd='numpy'):
        r"""Build a low rank risk model from past returns that include NaNs.

        This is an experimental procedure that may work well on past
        returns matrices with few NaN values (say, below 20% of the
        total entries). If there are (many) NaNs, one should probably
        also use a rather large risk forecast error.

        :param rets: Past returns, excluding cash.
        :type rets: pandas.DataFrame
        :param num_factors: How many factors in the fitted model.
        :type num_factors: int
        :param iters: How many iterations of SVD are performed.
        :type iters: int
        :param svd: Which singular value decomposition routine is used,
            default (and currently the only one supported) is ``'numpy'``.
        :type svd: str

        :raises SyntaxError: If wrong ``svd`` parameter is supplied.
        :raises cvxportfolio.errors.ForecastError: If the procedure fails;
             you may try with lower ``num_factors`` or ``iters``.

        :returns: (F, d)
        :rtype: (numpy.array, numpy.array)
        """
        nan_fraction = rets.isnull().sum().sum() / np.prod(rets.shape)
        normalizer = np.sqrt((rets**2).mean())
        normalized = rets
        if nan_fraction:
            nan_implicit_imputation = pd.DataFrame(0.,
                columns=normalized.columns, index = normalized.index)
            for _ in range(iters):
                if svd == 'numpy':
                    u, s, v = np.linalg.svd(
                        normalized.fillna(nan_implicit_imputation),
                        full_matrices=False)
                else:
                    raise SyntaxError(
                        'Currently only numpy svd is implemented')
                nan_implicit_imputation = pd.DataFrame(
                    (u[:, :num_factors] * (s[:num_factors]
                        )) @ v[:num_factors],
                    columns = normalized.columns, index = normalized.index)
        else:
            if svd == 'numpy':
                u, s, v = np.linalg.svd(normalized, full_matrices=False)
            else:
                raise SyntaxError(
                    'Currently only numpy svd is implemented')
        F = v[:num_factors].T * s[:num_factors] / np.sqrt(len(rets))
        F = pd.DataFrame(F.T, columns=normalizer.index)
        idyosyncratic = normalizer**2 - (F**2).sum(0)
        if not np.all(idyosyncratic >= 0.):
            raise ForecastError(
                "Low rank risk estimation with iterative SVD did not work."
                + " You probably have too many missing values in the past"
                + " returns. You may try with HistoricalFactorizedCovariance,"
                + " or change your universe.")
        return F.values, idyosyncratic.values

    # Will be exposed to the user, for now it's a class level constant
    _CACHED = True

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Current low-rank model, also cached.

        :param past_returns: Past market returns (including cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Extra arguments passed to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Low-rank plus diagonal covariance model: (F, d); excludes
             cash.
        :rtype: (numpy.array, numpy.array)
        """
        return self.build_low_rank_model(past_returns.iloc[:, :-1],
            num_factors=self.num_factors,
            iters=self.svd_iters, svd=self.svd)
