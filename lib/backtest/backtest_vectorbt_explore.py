# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:07:42 2024

@author: awei
"""

first_valid_index = data.apply(lambda col: col.first_valid_index())

# Create a boolean mask for valid data
valid_data_mask = data.apply(lambda col: col.index >= first_valid_index[col.name])





portfolio_daily_df1[['SH.159001','SH.159003']].count()

position_mask
segment_mask 

portfolio.orders.count()
Out[50]: 
symbol
SH.159001    1
SH.159003    1
Name: count, dtype: int32


portfolio.total_return(segment_mask=segment_mask)
portfolio.value()







d1=vbt.Portfolio.from_signals(data[['SH.159001', 'SH.159003']],fillna_close=False).stats(agg_func=None)
d1.loc['SH.159001',:]



vbt.Portfolio.from_signals(data[['SH.159001', 'SH.159003']],fillna_close=False).stats(agg_func=None).loc['SH.159001',:]

Out[97]: 
Start                         2010-01-04 00:00:00
End                           2021-12-31 00:00:00
Period                                       2917
Start Value                                 100.0
End Value                                 100.002
Total Return [%]                            0.002
Benchmark Return [%]                          0.0
Max Gross Exposure [%]                      100.0
Total Fees Paid                               0.0
Max Drawdown [%]                         0.316949
Max Drawdown Duration                      1673.0
Total Trades                                    1
Total Closed Trades                             0
Total Open Trades                               1
Open Trade PnL                              0.002
Win Rate [%]                                  NaN
Best Trade [%]                                NaN
Worst Trade [%]                               NaN
Avg Winning Trade [%]                         NaN
Avg Losing Trade [%]                          NaN
Avg Winning Trade Duration                    NaN
Avg Losing Trade Duration                     NaN
Profit Factor                                 NaN
Expectancy                                    NaN



Out[99]: 
Start                    2010-01-04 00:00:00
End                      2021-12-31 00:00:00
Period                                  2917
Total Return [%]                       0.002
Benchmark Return [%]                   0.002
Max Drawdown [%]                    0.316949
Max Drawdown Duration                 1673.0
Skew                               -5.299359
Kurtosis                          303.878071
Tail Ratio                           1.00001
Value at Risk                       -0.00003
Beta                                     1.0
Name: SH.159001, dtype: object