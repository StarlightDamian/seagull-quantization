# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:50:06 2024

@author: awei
(backtest_analyze)
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt

from __init__ import path
from base import base_connect_database

date_start='2019-01-01'
date_end='2023-01-01'
with base_connect_database.engine_conn('postgre') as conn:
    #data = pd.read_sql("dwd_freq_full_portfolio_daily_backtest", con=conn.engine)
    data = pd.read_sql(f"SELECT * FROM dwd_freq_full_portfolio_daily_backtest WHERE date BETWEEN '{date_start}' AND '{date_end}'", con=conn.engine)

data.index = data.date
#data.index = pd.to_datetime(data.index)

data = data[['SH.512690', 'SH.510300']]    #'SH.513360', 
macd = vbt.MACD.run(
    data,
    fast_window=12,  # Fast EMA period, default value 12
    slow_window=26,  # Slow EMA period, default value 26
    signal_window=9,  # Signal line period, default value 9
    macd_ewm=False,
    signal_ewm=False,
    adjust=False
)

# https://github.com/polakowo/vectorbt/issues/136
entries = macd.macd_above(0) & macd.macd_below(0).vbt.signals.fshift(1)
exits = macd.macd_below(0) & macd.macd_above(0).vbt.signals.fshift(1)

# =============================================================================
# portfolio_params={'freq': 'd',
#                  'fees': 0.001,  # 0.1% per trade
#                  'slippage': 0.001,  # 0.1% slippage
#                  'init_cash': 10000}
# =============================================================================
portfolio_params={'freq': 'd',
                 'fees': 0.001,  # 0.1% per trade
                 'slippage': 0.001,  # 0.1% slippage
                 'init_cash': 10000,
                 }
portfolio_strategy = vbt.Portfolio.from_signals(data,
                                       entries.astype(np.bool_),
                                       exits.astype(np.bool_),
                                       # ffill_val_price=True,
                                       **portfolio_params
                                       )
#portfolio_strategy.stats(agg_func=None)
metrics_strategy_df = portfolio_strategy.returns_stats(agg_func=None)
print(metrics_strategy_df['Annualized Return [%]'])

# =============================================================================
# data.index=data.date
# data['date_start'] = data.apply(lambda col: col.first_valid_index()).tolist()
# data['date_end'] = data.apply(lambda col: col.last_valid_index()).tolist()
#         
# data['price_start'] = data.apply(lambda col: col.loc[col.first_valid_index()])
# data['price_end'] = data.apply(lambda col: col.loc[col.last_valid_index()])
# =============================================================================
print(data.apply(lambda col: col.loc[col.first_valid_index()]))
print(data.apply(lambda col: col.loc[col.last_valid_index()]))

portfolio_base = vbt.Portfolio.from_holding(data,
                                            # ffill_val_price=True,
                                            **portfolio_params)
#portfolio_base.stats(agg_func=None)
metrics_base_df = portfolio_base.returns_stats(agg_func=None,
                                               year_freq='243 days',
                                               )
print(metrics_base_df['Annualized Return [%]'])
#print(metrics_base_df['Total Return [%]'])

trades_records_readable_strategy_df = portfolio_strategy.trades.records_readable
#https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/portfolio/base.py#L3622
orders_records_readable_strategy_df = portfolio_strategy.orders.records_readable
trades_records_readable_strategy_df.to_csv(f'{path}/data/trades_records_readable.csv',index=False)
orders_records_readable_strategy_df.to_csv(f'{path}/data/orders_records_readable.csv',index=False)
#['Exit Trade Id', 'Column', 'Size', 'Entry Timestamp', 'Avg Entry Price',
#       'Entry Fees', 'Exit Timestamp', 'Avg Exit Price', 'Exit Fees', 'PnL',
#       'Return', 'Direction', 'Status', 'Position Id']

fig = portfolio_strategy[(12, 26, 9, False, False,'SH.512690')].plot()
fig.write_html(f"{path}/plt/portfolio_strategy.html")

# 分别计算每年的
portfolio = vbt.Portfolio.from_holding(
    close=data,  # 你的收盘价数据
    init_cash=100_000  # 初始资金
)

returns_series = portfolio.returns()
annual_returns = returns_series.resample('YE').apply(lambda x: (1 + x).prod() - 1)
print(annual_returns)




 
