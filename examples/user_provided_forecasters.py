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
"""Simple example for providing user-defined forecasters to Cvxportfolio.

This example shows how the user can provide custom-made predictors for expected
returns and covariances, at each point in time (when used in back-test). These
forecasters can be used seamlessly inside a cvxportfolio back-test, or online
execution.

.. note::

    No attempt is being made here to find good values of the hyper-parameters;
    this is simply used to show how to provide custom forecasters.

One interesting feature of this is that Cvxportfolio **guarantees** that no
look-ahead biases are introduced when making these forecasts. The data is
sliced so that, at each point in time, only past data is provided to the
forecasters.

More advanced custom predictors can be provided as well; the relevant
interfaces will be documented in the future. Internally, Cvxportfolio
forecasters use something very similar to this, but also have a recursive
execution model which enables them to compose objects, are aware of the current
trading universe and future (expected) trading calendar (for multi-period
applications), and have a destructor which is used for memory safety when
parallelizing back-tests. However, this simple interface is guaranteed to work.

You can see the documentation of the
:meth:`cvxportfolio.estimator.Estimator.values_in_time` method that is used
here for the full list of available arguments.
"""

import os

import matplotlib.pyplot as plt

if __name__ == '__main__':

    import cvxportfolio as cvx

    # Here we define a class to forecast expected returns
    # There is no need to inherit from a base class, in this simple case
    class WindowMeanReturns: # pylint: disable=too-few-public-methods
        """Expected return as mean of recent window of past returns.

        This is only meant as an example of how to define a custom forecaster;
        it is not very interesting. Since version ``1.2.0`` a similar
        functionality has been included in the default forecasters classes.

        :param window: Window used for the mean returns.
        :type window: int
        """

        def __init__(self, window=20):
            self.window = window

        def values_in_time(self, past_returns, **kwargs):
            """This method computes the quantity of interest.

            It has many arguments, we only need to use ``past_returns`` in this
            case.

            :param past_returns: Historical market returns for all assets in
                the current trading universe, up to each time at which the
                policy is evaluated.
            :type past_returns: pd.DataFrame
            :param kwargs: Other, unused, arguments to :meth:`values_in_time`.
            :type kwargs: dict

            :returns: Estimated mean returns.
            :rtype: pd.Series

            .. note::

                The last column of ``past_returns`` are the cash returns.
                You need to explicitely skip them otherwise Cvxportfolio will
                throw an error.
            """
            return past_returns.iloc[-self.window:, :-1].mean()

    # Here we define a class to forecast covariances
    # There is no need to inherit from a base class, in this simple case
    class WindowCovariance: # pylint: disable=too-few-public-methods
        """Covariance computed on recent window of past returns.

        This is only meant as an example of how to define a custom forecaster;
        it is not very interesting. Since version ``1.2.0`` a similar
        functionality has been included in the default forecasters classes.

        :param window: Window used for the covariance computation.
        :type window: int
        """

        def __init__(self, window=20):
            self.window = window

        def values_in_time(self, past_returns, **kwargs):
            """This method computes the quantity of interest.

            It has many arguments, we only need to use ``past_returns`` in this
            case.

            :param past_returns: Historical market returns for all assets in
                the current trading universe, up to each time at which the
                policy is evaluated.
            :type past_returns: pd.DataFrame
            :param kwargs: Other, unused, arguments to :meth:`values_in_time`.
            :type kwargs: dict

            :returns: Estimated covariance.
            :rtype: pd.DataFrame

            .. note::

                The last column of ``past_returns`` are the cash returns.
                You need to explicitely skip them otherwise Cvxportfolio will
                throw an error.
            """
            return past_returns.iloc[-self.window:, :-1].cov()

    # define the hyper-parameters
    WINDOWMU = 252
    WINDOWSIGMA = 252
    GAMMA_RISK = 5
    GAMMA_TRADE = 3

    # define the forecasters
    mean_return_forecaster = WindowMeanReturns(WINDOWMU)
    covariance_forecaster = WindowCovariance(WINDOWSIGMA)

    # define the policy
    policy = cvx.SinglePeriodOptimization(
        objective = cvx.ReturnsForecast(r_hat = mean_return_forecaster)
            - GAMMA_RISK * cvx.FullCovariance(Sigma = covariance_forecaster)
            - GAMMA_TRADE * cvx.StocksTransactionCost(),
        constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
        )

    # define the simulator
    simulator = cvx.StockMarketSimulator(['AAPL', 'GOOG', 'MSFT', 'AMZN'])

    # back-test
    result = simulator.backtest(policy, start_time='2020-01-01')

    # show the result
    print(result)
    figure = result.plot()

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        figure.savefig('user_provided_forecasters.png')
    else:
        plt.show()
