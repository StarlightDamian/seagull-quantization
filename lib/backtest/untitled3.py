# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:02:53 2024

@author: awei
"""

import vectorbt as vbt
import pandas as pd
from multiprocessing import Pool, cpu_count

def strategy(data, window_fast=10, window_slow=50):
    # Define simple moving average (SMA)
    #fast = vbt.MA.run(data, window=window_fast)
    #slow = vbt.MA.run(data, window=window_slow)
    # Generate buy and sell signals
    print(window_fast,window_slow)
    entries = window_fast
    exits = window_slow
    return entries, exits

def compute_strategy(args):
    data, params = args
    return strategy(data, **params)

def parallel_strategy(data, strategy_params_list):
    # Create a list of arguments for each strategy computation
    args_list = [(data, params) for params in strategy_params_list]
    
    # Use multiprocessing to parallelize the computation
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_strategy, args_list)
    
    # Combine the results
    all_entries = pd.DataFrame({f"strategy_{i}": res[0] for i, res in enumerate(results)})
    all_exits = pd.DataFrame({f"strategy_{i}": res[1] for i, res in enumerate(results)})
    
    return all_entries, all_exits

# Usage
data = ...  # Your price data
strategy_params_list = [{'window_fast': 10, 'window_slow': 50}, {'window_fast': 20, 'window_slow': 100}, ...]
entries, exits = parallel_strategy(data, strategy_params_list)