Back-test result 
================

.. py:module:: cvxportfolio
     :noindex:

.. automodule:: cvxportfolio.result

.. autoclass:: BacktestResult

    .. automethod:: plot

    .. automethod:: times_plot

    .. autoproperty:: logs

    .. autoproperty:: cash_key
    
    .. autoproperty:: periods_per_year
    
    .. autoproperty:: v
    
    .. autoproperty:: profit
    
    .. autoproperty:: w
    
    .. autoproperty:: h_plus
    
    .. autoproperty:: w_plus
    
    .. autoproperty:: leverage
    
    .. autoproperty:: turnover
    
    .. autoproperty:: returns
    
    .. autoproperty:: average_return
    
    .. autoproperty:: annualized_average_return
    
    .. autoproperty:: growth_rates
    
    .. autoproperty:: average_growth_rate
    
    .. autoproperty:: annualized_average_growth_rate
    
    .. autoproperty:: volatility
    
    .. autoproperty:: annualized_volatility
    
    .. autoproperty:: quadratic_risk
    
    .. autoproperty:: annualized_quadratic_risk
    
    .. autoproperty:: excess_returns

    .. autoproperty:: excess_volatility
    
    .. autoproperty:: average_excess_return
    
    .. autoproperty:: annualized_average_excess_return
    
    .. autoproperty:: annualized_excess_volatility
    
    .. autoproperty:: active_returns

    .. autoproperty:: active_volatility
    
    .. autoproperty:: average_active_return
    
    .. autoproperty:: annualized_average_active_return
    
    .. autoproperty:: annualized_active_volatility
    
    .. autoproperty:: sharpe_ratio
    
    .. autoproperty:: information_ratio
    
    .. autoproperty:: excess_growth_rates

    .. autoproperty:: average_excess_growth_rate

    .. autoproperty:: annualized_average_excess_growth_rate
    
    .. autoproperty:: active_growth_rates

    .. autoproperty:: average_active_growth_rate

    .. autoproperty:: annualized_average_active_growth_rate
    
    .. autoproperty:: drawdown
    
    .. autoproperty:: policy_times

    .. autoproperty:: simulator_times

    .. autoproperty:: market_data_times

    .. autoproperty:: result_times

Interface methods with the market simulator
-------------------------------------------

.. autoclass:: BacktestResult
     :noindex:

     .. automethod:: log_trading

     .. automethod:: log_final