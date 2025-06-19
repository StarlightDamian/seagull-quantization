# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:12:38 2025

@author: awei
demo_test
"""
from seagull.settings import PATH

import pandas as pd
from feature import vwap, max_drawdown

def bar_high(raw_df, window=10):
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['high','close']]
    df['y_high'] = df['high'].rolling(window=window, min_periods=1).max()
    df['y_high'] = df['y_high'].shift(-window+1)
    raw_df['y_high_rate'] = df['y_high'].div(df['close'], axis=0)
    return raw_df

def bar_low(raw_df, window=10):
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['low','close']]
    df['y_low'] = df['low'].rolling(window=window, min_periods=1).min()
    df['y_low'] = df['y_low'].shift(-window+1)
    raw_df['y_low_rate'] = df['y_low'].div(df['close'], axis=0)
    return raw_df

def n_day_pred(raw_df, window=10, freq='d'):
    freq = f'{window}{freq}'
    
    # N日平均成本
    raw_df = raw_df.groupby('full_code').apply(vwap.daily_vwap, window=window)
    raw_df = raw_df.rename(columns={'y_vwap_rate': f'y_{window}d_vwap_rate'}).reset_index(drop=True)
    
    # N日最大回撤
    raw_df = raw_df.groupby('full_code').apply(max_drawdown.calculate_max_drawdown, window=window)
    raw_df = raw_df.rename(columns={'y_max_dd': f'y_{window}d_max_dd'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','y_10d_max_dd']]
    
    # N日最高价
    raw_df = raw_df.groupby('full_code').apply(bar_high, window=window)
    raw_df = raw_df.rename(columns={'y_high_rate': f'y_{window}d_high_rate'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','high','y_10d_high','y_10d_high_rate']]
    
    # N日最低价
    raw_df = raw_df.groupby('full_code').apply(bar_low, window=window)
    raw_df = raw_df.rename(columns={'y_low_rate': f'y_{window}d_low_rate'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','low','y_10d_low','y_10d_low_rate']]
    
    # VWAP-to-Drawdown Ratio
    raw_df[f'y_{window}d_vwap_drawdown_rate'] = raw_df[f'y_{window}d_vwap_rate'] / (raw_df[f'y_{window}d_max_dd'] + 1)
    return raw_df
    
    
if __name__ == '__main__':
    from seagull.utils import utils_database
    
    with utils_database.engine_conn("POSTGRES") as conn:
        raw_df = pd.read_sql('select date, full_code, high, low, close, volume from "dwd_ohlc_incr_stock_daily-bak" limit 100000', con=conn.engine)
        
    # 计算每个日期的最大回撤，使用窗口10天
    window=3
    raw_df = n_day_pred(raw_df, window=window, freq='d')
    raw_df.to_csv(f'{PATH}/_file/test_vwap_drawdown_rate.csv',index=False)
    # raw_df[['y_3d_vwap_rate','y_3d_max_dd','y_3d_vwap_drawdown_rate']]
    # raw_df[f'y_{window}d_vwap_drawdown_rate'] = raw_df[f'y_{window}d_vwap_rate'] / raw_df[f'y_{window}d_max_dd']
    # raw_df[f'y_3d_vwap_drawdown_rate'] = raw_df[f'y_3d_vwap_rate'] / raw_df[f'y_3d_max_dd']
    
    # 输出过去100天的最大回撤
    # print(max_drawdown.tail(100))  # 输出最后100天的最大回撤