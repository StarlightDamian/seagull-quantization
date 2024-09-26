# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:17:57 2024

@author: awei
backtest_vectorbt_matrix
"""
import vectorbt as vbt
import numpy as np
import pandas as pd

def batch_backtest(symbols, start_date, end_date):
    # Download data for all symbols at once
    data = vbt.YFData.download(symbols, start=start_date, end=end_date, missing_index='drop').get('Close')
    
    # Calculate SMAs for all symbols at once
    fast_ma = vbt.MA.run(data, window=10).ma
    slow_ma = vbt.MA.run(data, window=50).ma
    
    # Ensure no NaN values and valid comparison
    fast_ma_clean = fast_ma.fillna(0)
    slow_ma_clean = slow_ma.fillna(0)
    #global fast_ma_clean1, slow_ma_clean1
    #fast_ma_clean1 = fast_ma_clean
    #slow_ma_clean1 = slow_ma_clean
    # Generate entry and exit signals for all symbols, convert to boolean
    #fast_ma_clean, slow_ma_clean = fast_ma_clean.align(slow_ma_clean, join='inner', axis=0)
    
    #  the difference in the MultiIndex 会导致无法进行大小比较
    fast_ma_clean = fast_ma_clean.droplevel('ma_window', axis=1)
    slow_ma_clean = slow_ma_clean.droplevel('ma_window', axis=1)

    entries = fast_ma_clean > slow_ma_clean
    exits = fast_ma_clean < slow_ma_clean
    
    # Run the portfolio simulation for all symbols at once
    portfolio = vbt.Portfolio.from_signals(
        data, 
        entries.astype(np.bool_),  # Ensure boolean values
        exits.astype(np.bool_),    # Ensure boolean values
        init_cash=10000,
        fees=0.001,  # 0.1% fees
        freq='1D'
    )
    
    # Get total return and Sharpe ratio for each symbol
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()

    # Combine results into a DataFrame
    results = pd.DataFrame({
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio
    })
    
    return results

# Example usage
if __name__ == '__main__':
    symbols = ['AAPL', 'MSFT', 'GOOG']  # List of stock symbols
    start_date = '2021-01-01'
    end_date = '2022-01-01'
    
    #results = batch_backtest(symbols, start_date, end_date)
    #print(results)
    data = vbt.YFData.download(symbols, start=start_date, end=end_date, missing_index='drop').get('Close')
    
    
    fast = vbt.MA.run(data, window=10)
    slow = vbt.MA.run(data, window=50)
    
    # Calculate SMAs for all symbols at once
    fast_ma = fast.ma
    slow_ma = slow.ma
    
    # Ensure no NaN values and valid comparison
    fast_ma_clean = fast_ma.fillna(0)
    slow_ma_clean = slow_ma.fillna(0)
    #global fast_ma_clean1, slow_ma_clean1
    #fast_ma_clean1 = fast_ma_clean
    #slow_ma_clean1 = slow_ma_clean
    # Generate entry and exit signals for all symbols, convert to boolean
    #fast_ma_clean, slow_ma_clean = fast_ma_clean.align(slow_ma_clean, join='inner', axis=0)
    
    #  the difference in the MultiIndex 会导致无法进行大小比较
    fast_ma_clean = fast_ma_clean.droplevel('ma_window', axis=1)
    slow_ma_clean = slow_ma_clean.droplevel('ma_window', axis=1)

    #entries = fast_ma_clean > slow_ma_clean
    #exits = fast_ma_clean < slow_ma_clean
    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)
    
    # Run the portfolio simulation for all symbols at once
    portfolio = vbt.Portfolio.from_signals(
        data, 
        entries.astype(np.bool_),  # Ensure boolean values
        exits.astype(np.bool_),    # Ensure boolean values
        init_cash=10000,
        fees=0.001,  # 0.1% fees
        freq='1D'
    )
    
    # Get total return and Sharpe ratio for each symbol
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    
    # Combine results into a DataFrame
    results = pd.DataFrame({
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio
    })
    #results = batch_backtest(symbols, start_date, end_date)
    print(results)
# =============================================================================
#             Total Return  Sharpe Ratio
#     symbol                            
#     AAPL        0.187136      1.075196
#     MSFT        0.266044      1.601834
#     GOOG        0.557217      2.483150
#     AMZN       -0.181926     -1.073387
# 
# =============================================================================
