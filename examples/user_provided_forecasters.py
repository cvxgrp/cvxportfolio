"""
This example shows how the user can provide custom-made
predictors for expected returns and covariances, 
at each point in time of the backtest. These can be 
used seamlessly inside a cvxportfolio backtest routine.
"""

import cvxportfolio as cvx


# Here we define a class to forecast expected returns
class WindowMeanReturn:
    """Expected return as mean of recent window of past returns."""
    
    def __init__(self, window=20):
        self.window = window
    
    def values_in_time(self, past_returns, **kwargs):
        """This method computes the quantity of interest.
        
        It has many arguments, we only need to use past_returns
        in this case. 
        
        NOTE: the last column of `past_returns` are the cash returns.
        You need to explicitely skip them otherwise the compiler will
        throw an error.
        """
        return past_returns.iloc[-self.window:, :-1].mean()
        
        
# define the policy
policy = cvx.SinglePeriodOptimization(
    objective = cvx.ReturnsForecast(WindowMeanReturn(250)) # window size 250
        - .5 * cvx.FullCovariance() # you can adjust these multipliers
        - 3 * cvx.StocksTransactionCost(), # you can adjust these multipliers
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
    )

# define the simulator 
simulator = cvx.StockMarketSimulator(['AAPL', 'GOOG', 'MSFT', 'AMZN'])

# backtest
result = simulator.backtest(policy, start_time='2020-01-01')

# show the result
print(result)
result.plot()