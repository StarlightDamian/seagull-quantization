# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:41:19 2024

@author: awei
"""

import vectorbt as vbt
import cudf
import cupy as cp
import pandas as pd

def optimized_pipeline_gpu(symbols, start_date, end_date):
    # Download data for all symbols at once (we'll use normal pandas for this)
    data = vbt.YFData.download(symbols, start=start_date, end=end_date, missing_index='drop').get('Close')
    
    # Convert pandas DataFrame to cuDF DataFrame to use GPU
    data_gpu = cudf.DataFrame.from_pandas(data)
    
    # Calculate SMAs for all symbols at once using cuDF
    fast_ma_gpu = data_gpu.rolling(window=10).mean()
    slow_ma_gpu = data_gpu.rolling(window=50).mean()

    # Convert cuDF back to cupy array for GPU-based matrix operations
    fast_ma_np = fast_ma_gpu.to_cupy()
    slow_ma_np = slow_ma_gpu.to_cupy()

    # Generate entry and exit signals using cupy arrays
    entries = cp.greater(fast_ma_np, slow_ma_np)
    exits = cp.less_equal(fast_ma_np, slow_ma_np)
    
    # Run the portfolio simulation for all symbols using GPU (data and signals are in cupy arrays)
    portfolio = vbt.Portfolio.from_signals(
        data,  # Use the original pandas DataFrame for vectorbt
        entries=cp.asnumpy(entries),  # Convert back to numpy for vectorbt
        exits=cp.asnumpy(exits),  # Convert back to numpy for vectorbt
        init_cash=10000,
        fees=0.001,  # 0.1% fees
        freq='1D'
    )
    
    # Get total return for each symbol
    total_return = portfolio.total_return()
    
    # Get Sharpe ratio for each symbol
    sharpe_ratio = portfolio.sharpe_ratio()
    
    # Combine results
    results = pd.DataFrame({
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio
    })
    
    return results

if __name__ == '__main__':
    # Generate 1000 random stock symbols
    symbols = [f'STOCK{i:04d}' for i in range(1000)]
    
    start_date = '2021-01-01'
    end_date = '2022-01-01'
    
    results = optimized_pipeline_gpu(symbols, start_date, end_date)
    print(results)
