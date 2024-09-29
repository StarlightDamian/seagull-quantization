# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:36:21 2024

@author: awei
"""
import torch
import vectorbt as vbt
import numpy as np
import pandas as pd
from loguru import logger
logger.add(
    sink=lambda msg: print(msg, end=''),  # 控制台输出
    format="{time:YY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="INFO"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Convert data to PyTorch tensors
def to_torch(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return torch.from_numpy(data.values).to(device)
    else:
        return torch.tensor(data, device=device)
data = vbt.YFData.download(["ADA-USD", "ETH-USD"]).get('Close')

pf = vbt.Portfolio.from_holding(data, init_cash=100)

fast_ma = vbt.MA.run(data, 10)
slow_ma = vbt.MA.run(data, 50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
logger.info('t1_start')
pf = vbt.Portfolio.from_signals(data, entries, exits, init_cash=100)
logger.info('t1_end')
# pf.stats()




# =============================================================================
# import numpy as np
# import pandas as pd
# import vectorbt as vbt
# 
# # Check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# 
# 
# 
# # Example usage with VectorBT
# data = pd.DataFrame(np.random.randn(1000, 4), columns=['A', 'B', 'C', 'D'])
# entries = np.random.choice([True, False], size=data.shape)
# exits = np.random.choice([True, False], size=data.shape)
# =============================================================================

# Convert to PyTorch tensors
data_torch = to_torch(data)
entries_torch = to_torch(entries)
exits_torch = to_torch(exits)

# Use with VectorBT
logger.info('t2_start')
portfolio = vbt.Portfolio.from_signals(
    data_torch.cpu(),  # Convert back to numpy for VectorBT
    entries_torch.cpu(),
    exits_torch.cpu()
)
# =============================================================================
# portfolio = vbt.Portfolio.from_signals(
#     data_torch.cpu().numpy(),  # Convert back to numpy for VectorBT
#     entries_torch.cpu().numpy().astype(bool),
#     exits_torch.cpu().numpy().astype(bool)
# )
# =============================================================================
logger.info('t2_end')
# Perform calculations on GPU
def calculate_returns(data):
    return (data[1:] / data[:-1]) - 1

returns = calculate_returns(data_torch)

# Convert back to numpy for further use with VectorBT if needed
returns_np = returns.cpu().numpy()



# =============================================================================
# 
# Out[9]: 
# Start                         2014-09-17 00:00:00+00:00
# End                           2024-09-18 00:00:00+00:00
# Period                                             3654
# Start Value                                       100.0
# End Value                                  23215.332515
# Total Return [%]                           23115.332515
# Benchmark Return [%]                       13142.175453
# Max Gross Exposure [%]                            100.0
# Total Fees Paid                                     0.0
# Max Drawdown [%]                              72.193683
# Max Drawdown Duration                            1053.0
# Total Trades                                         42
# Total Closed Trades                                  42
# Total Open Trades                                     0
# Open Trade PnL                                      0.0
# Win Rate [%]                                       50.0
# Best Trade [%]                               333.139712
# Worst Trade [%]                              -19.973111
# Avg Winning Trade [%]                         55.093046
# Avg Losing Trade [%]                          -8.898233
# Avg Winning Trade Duration                    80.095238
# Avg Losing Trade Duration                     16.619048
# Profit Factor                                  2.303352
# Expectancy                                    550.36506
# dtype: object
# 
# Out[14]: 
# Start                    2014-09-17 00:00:00+00:00#
# End                      2024-09-18 00:00:00+00:00#
# Period                                        3654
# Total Return [%]                      23115.332515
# Benchmark Return [%]                  13155.324861
# Max Drawdown [%]                         72.193683
# Max Drawdown Duration                       1053.0
# Skew                                      0.429604
# Kurtosis                                 10.824185
# Tail Ratio                                1.210349
# Value at Risk                            -0.037259
# Beta                                      0.533204
# dtype: object
# =============================================================================
