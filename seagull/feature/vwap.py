# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:09:15 2024

@author: awei
vwap

VWAP（**Volume-Weighted Average Price**）成交量加权平均价是一个衡量某段时间内加权平均价格的指标，常用于衡量一个证券在特定交易日内的平均交易价格，权重通常是交易量。它是一种广泛使用的技术分析工具，特别是在美股和A股市场中，帮助交易者评估股票在某一天的价格水平。
vwap = ∑(成交价 * 成交量) / ∑(成交量)
"""
import pandas as pd
import numpy as np


def minute_vwap(df, bar='1T', default_price_col='close', default_vol_col='volume'):
    """
    计算按分钟频率的 VWAP

    参数:
    - df: 包含价格和成交量的DataFrame，需有时间戳、价格和成交量
    - bar: 计算时间粒度（如：'1T' 表示1分钟，'1D' 表示按天计算）
    - default_price_col: 默认的价格列名
    - default_vol_col: 默认的成交量列名

    返回:
    - vwap: 各时间段的 VWAP 值
    """

    # 转换时间戳为 pandas 的 datetime 类型（假设已经是datetime类型）
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 按时间频率重新采样数据
    df_resampled = df.resample(bar, on='timestamp').agg({default_price_col: 'ohlc', default_vol_col: 'sum'})
    
    # 计算加权平均价格
    vwap = (df_resampled[('close', 'close')] * df_resampled[default_vol_col]).cumsum() / df_resampled[default_vol_col].cumsum()

    return vwap


def daily_vwap(raw_df, window=20, default_price_col='close', default_vol_col='volume'):
    """
    计算按窗口大小（20日）的 VWAP

    参数:
    - df: 包含价格和成交量的DataFrame，需有时间戳、价格和成交量
    - window: 计算VWAP的窗口大小，默认为20（即过去20个交易日）
    - default_price_col: 默认的价格列名
    - default_vol_col: 默认的成交量列名

    返回:
    - vwap: 各天的 VWAP 值
    """

    # 确保数据按日期排序
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['date']+[default_price_col]+[default_vol_col]]
    #df['date'] = pd.to_datetime(df['date'])
    #df = df.sort_values(by='date')
    
    n_day_vwap = (df[default_price_col] * df[default_vol_col]).rolling(window=window).sum() / df[default_vol_col].rolling(window=window).sum()
    df['vwap'] = n_day_vwap.fillna(np.nan)
    df['y_vwap'] = df['vwap'].shift(-window+1)
    raw_df['y_vwap_rate'] =  df['y_vwap'].div(df['close'], axis=0)
    return raw_df

    # 计算按20日滑动窗口的VWAP
    #df['vwap'] = (df[default_price_col] * df[default_vol_col]).rolling(window=window).sum() / df[default_vol_col].rolling(window=window).sum()
    #return df

#def vwap_pipeline(raw_df, column_name, window=10):
#    df = daily_vwap(raw_df, window=window)
#    num = (window-1)
#    df[column_name] = df.vwap.shift(-num)
#    df = df.head(-num)
#    return df
    
if __name__ == '__main__':
    from __init__ import path
    from utils import utils_database
    
    with utils_database.engine_conn('postgre') as conn:
        raw_df = pd.read_sql("select date, full_code, close, volume from dwd_ohlc_incr_stock_daily where full_code='510300.sh'", con=conn.engine)
    
    window=3
    raw_df = daily_vwap(raw_df, window=window)
    raw_df = raw_df.rename(columns={'y_vwap_rate': f'y_{window}d_vwap_rate'})
    
    #df = vwap_pipeline(raw_df, 
    #                   column_name='y_10d_vwap',
    #                   window=10)
        
    # 示例数据
# =============================================================================
#     data = {
#         'timestamp': ['2022-09-20 09:00:00', '2022-09-20 09:01:00', '2022-09-20 09:02:00', '2022-09-20 09:03:00'],
#         'close': [100, 101, 102, 103],
#         'volume': [200, 300, 400, 500]
#     }
#     df = pd.DataFrame(data)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     
#     # 计算分钟级别的VWAP
#     minute_vwap_result = minute_vwap(df)
#     print(minute_vwap_result)
#     
#     # 计算日级别的VWAP
#     daily_vwap_result = daily_vwap(df)
#     print(daily_vwap_result)
# =============================================================================
