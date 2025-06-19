# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:17:50 2025

@author: awei
max_drawdown
"""

import pandas as pd
import numpy as np

def calculate_max_drawdown(raw_df, column_name='10d_max_dd', window=10):
    """
    计算最大回撤。
    
    参数：
    - df: 包含股票收盘价的DataFrame，索引为日期，列为股票代码
    - window: 计算最大回撤的滚动窗口大小，默认为10
    
    返回：
    - max_drawdown: 每个日期的最大回撤
    """
    #df = raw_df.close
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['date','close']]

    
    # 计算滚动窗口内的最大值和最小值
    rolling_max = df['close'].rolling(window=window, min_periods=1).max()
    drawdowns = (df['close'] - rolling_max) / rolling_max  # 回撤比例
    
    # 返回每个日期的最大回撤
    # max_drawdown = drawdowns.min(axis=1)  # 每天的最大回撤值（每个日期的最大回撤）
    df['max_dd'] = abs(drawdowns)
    raw_df['y_max_dd'] = df['max_dd'].shift(-window+1)
    return raw_df


def calculate_max_recovery(df, column_name='y_max_recovery', window=10):
    """
    计算窗口期内的最大反弹（从最低点反弹到后续高点的最大涨幅）。
    
    参数：
    - df: 包含股票收盘价的DataFrame，需包含日期和收盘价
    - window: 滚动窗口大小，默认为10（单位与数据频率一致，如5分钟K线）
    - column_name: 输出列名
    
    返回：
    - df: 新增最大反弹列的DataFrame
    """
    df = df.sort_values(by='date')
    close = df['close'].values
    
    # 初始化结果数组
    max_recovery = np.zeros(len(close))
    
    for i in range(len(close)):
        if i < window:
            max_recovery[i] = 0.0
            continue
        
        # 窗口内的数据切片
        window_data = close[i - window : i]
        
        # 找到窗口内的最低点位置
        min_idx = np.argmin(window_data)
        min_price = window_data[min_idx]
        
        # 在最低点之后的子窗口中寻找最高点
        if min_idx < len(window_data) - 1:
            recovery_window = window_data[min_idx:]
            max_price = np.max(recovery_window)
            recovery = (max_price - min_price) / min_price
        else:
            recovery = 0.0
        
        max_recovery[i] = recovery
    
    df[column_name] = max_recovery
    return df


if __name__ == '__main__':
    from __init__ import path
    from utils import utils_database
    
    with utils_database.engine_conn('postgre') as conn:
        raw_df = pd.read_sql("select date, full_code, high, low, close, volume from dwd_ohlc_incr_stock_daily where full_code='510300.sh'", con=conn.engine)
        
    # 计算每个日期的最大回撤，使用窗口10天
    window=3
    raw_df = calculate_max_recovery(raw_df, window=window)
    raw_df = raw_df.rename(columns={'y_max_recovery': f'y_{window}d_max_recovery'})
    
# =============================================================================
#     raw_df['10d_high'] = raw_df['high'].rolling(window=window, min_periods=1).max()
#     raw_df['y_10d_high'] = raw_df['10d_high'].shift(-window+1)
#     raw_df['y_10d_high_rate'] = raw_df['y_10d_high'].div(raw_df['close'], axis=0)
#     
# =============================================================================
    
    # 输出过去100天的最大回撤
    #print(max_drawdown.tail(100))  # 输出最后100天的最大回撤
