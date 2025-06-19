# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:29:31 2025

@author: awei

把五分钟k线的转化为每日高频特征(dwd_feat_incr_high_frequency_5minute)

time(48 bar)
high, low, close, volume
date, code,

primary_key
date
time
code
open
high
low
close
volume
turnover

primary_key
date
full_code
open
5pct_5min_high
5pct_5min_low
10pct_5min_high
10pct_5min_low
5min_vwap
volume
turnover
"""
import os
import argparse

import numpy as np
import pandas as pd
# from joblib import Parallel, delayed

from __init__ import path
from utils import utils_database, utils_log, utils_data, utils_character, utils_thread, utils_time
from data import utils_api_baostock

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

def calculate_high_frequency(day_df, buy_pct=[10, 20], sell_pct=[80, 90]):
    day_df = day_df.reset_index(drop=True)
    total_volume = day_df['volume'].sum()
    ohlc_pct_columns = []
    
    # 计算买入价格（低价百分位）
    day_df = day_df.sort_values('low')
    day_df['low_cum_volume'] = day_df['volume'].cumsum()
    
    for buy_pct_1 in buy_pct:
        # 使用searchsorted高效查找百分位的位置,计算相应的买入
        buy_index = np.searchsorted(day_df['low_cum_volume'], (buy_pct_1 / 100) * total_volume)
        pct_low = f'_{buy_pct_1}pct_5min_low'
        if (buy_index < len(day_df)) and (total_volume!=0):
            day_df[pct_low] = day_df.iloc[buy_index]['low']
        else:
            # Handle case where index is out of bounds, you can choose to set NaN or the last valid value
            day_df[pct_low] = np.nan
        ohlc_pct_columns.append(pct_low)
        
    # 计算卖出价格（高价百分位）
    day_df = day_df.sort_values('high', ascending=False)
    day_df['high_cum_volume'] = day_df['volume'].cumsum()
    
    for sell_pct_1 in sell_pct:
        sell_index = np.searchsorted(day_df['high_cum_volume'], (1 - (sell_pct_1 / 100)) * total_volume)
        pct_high = f'_{sell_pct_1}pct_5min_high'
        if (sell_index < len(day_df)) and (total_volume!=0):
            day_df[pct_high] = day_df.iloc[sell_index]['high']
        else:
            # Handle case where index is out of bounds, you can choose to set NaN or the last valid value
            day_df[pct_high] = np.nan
        ohlc_pct_columns.append(pct_high)
        
    # vwap
    if total_volume==0:
        day_df['_5min_vwap'] = np.nan
    else:
        day_df['_5min_vwap'] = (((day_df.high + day_df.low + day_df.close) / 3) * day_df.volume).sum() / total_volume
        day_df['_5min_vwap'] = day_df['_5min_vwap'].round(4)
    
    day_df = day_df[['date', 'full_code', '_5min_vwap', 'primary_key', 'freq', 'adj_type'] + ohlc_pct_columns].head(1)
    #day_df = day_df.drop_duplicates('date', keep='first')
    return day_df

def pipeline(_5min_df):
    _5min_df['freq'] = 'd'
    _5min_df['adj_type'] = 'pre'
    _5min_df['_time'] = pd.to_datetime(_5min_df['date']).dt.strftime('%Y%m%d%H%M%S')

    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    _5min_df['primary_key'] = (_5min_df['_time'].astype(str) +
                               _5min_df['full_code'].astype(str) +
                               _5min_df['freq'].astype(str) +
                               _5min_df['adj_type'].astype(str)
                               ).apply(utils_character.md5_str) # md5（时间、带后缀代码、频率、复权）
    
    # 使用 ThreadPoolExecutor 并行处理
    grouped = _5min_df.groupby('primary_key')
    daily_df = utils_thread.thread(grouped,
                                   calculate_high_frequency,
                                   buy_pct=[5, 10, 15],
                                   sell_pct=[85, 90, 95],
                                   max_workers=8)
    #daily_df = _5min_df.groupby('primary_key').apply(calculate_buy_sell_prices, buy_pct=[10, 20], sell_pct=[80, 90])
    
    utils_data.output_database_large(daily_df,
                                     filename='dwd_feat_incr_high_frequency_5minute',
                                     if_exists='append',
                                     )

def dwd_baostock_minute(subtable):
    date = str(subtable.name)
    date_start = utils_time.date_plus_days(date, days=1)
    date_end = utils_time.date_plus_days(date, days=40)  # 和freq='20d'匹配
    logger.info(f'date_start: {date_start} | date_end:{date_end}')
    
    # 获取日期段数据
    with utils_database.engine_conn('postgre') as conn:
        asset_5min_df = pd.read_sql(f"SELECT date, time, code, high, low, close, volume FROM ods_ohlc_incr_baostock_stock_sh_sz_minute WHERE date >= '{date_start}' AND date < '{date_end}'", con=conn.engine)
    logger.info(f'asset_5min_df.shape: {asset_5min_df.shape}')
    
    asset_5min_df['volume'] = asset_5min_df['volume'].astype('int64')
    asset_5min_df[['high', 'low', 'close']] = asset_5min_df[['high', 'low', 'close']].astype(float).round(4)
    asset_5min_df['time'] = asset_5min_df['time'].str.slice(start=0, stop=14)
    asset_5min_df = utils_api_baostock.split_baostock_code(asset_5min_df)
    asset_5min_df = asset_5min_df[['date', 'time', 'high', 'low', 'close', 'volume', 'full_code']]
    #return asset_5min_df
    pipeline(asset_5min_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2019-01-01', help='When to start high frequency')
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='When to start high frequency')
    parser.add_argument('--date_end', type=str, default='2025-02-23', help='End time for high frequency')
    args = parser.parse_args()
    
    logger.info(f"""task: dwd_feat_incr_high_frequency_5minute
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")

    daily_dates = pd.date_range(start=args.date_start, end=args.date_end, freq='40d').strftime('%Y-%m-%d')
    daily_dates_df = pd.DataFrame(daily_dates, columns=['date'])
    daily_dates_df.groupby('date').apply(dwd_baostock_minute)
    
    
# =============================================================================
# def build_features(df_5min, lookback_days=5):
#     """
#     构建每日特征：
#     - 过去N日的波动率、成交量均值、价格动量等
#     - 日内特征：如振幅、VWAP（成交量加权平均价）
#     """
#     # 按日聚合计算基础指标
#     daily_open = df_5min.resample('D').first()['open']
#     daily_high = df_5min.resample('D').max()['high']
#     daily_low = df_5min.resample('D').min()['low']
#     daily_close = df_5min.resample('D').last()['close']
#     daily_volume = df_5min.resample('D').sum()['volume']
#     
#     # 计算VWAP（成交量加权平均价）
#     df_5min['vwap'] = (df_5min['volume'] * (df_5min['high'] + df_5min['low'] + df_5min['close']) / 3).cumsum() / df_5min['volume'].cumsum()
#     daily_vwap = df_5min.resample('D').last()['vwap']
#     
#     # 合并基础特征
#     features = pd.DataFrame({
#         'open': daily_open,
#         'high': daily_high,
#         'low': daily_low,
#         'close': daily_close,
#         'volume': daily_volume,
#         'vwap': daily_vwap
#     })
#     
#     # 添加滚动窗口特征
#     for window in [3, 5, 10]:
#         features[f'volatility_{window}d'] = daily_close.pct_change().rolling(window).std()
#         features[f'volume_ma_{window}d'] = daily_volume.rolling(window).mean()
#     
#     # 添加滞后特征
#     for lag in [1, 2, 3]:
#         features[f'close_lag{lag}'] = daily_close.shift(lag)
#     
#     return features.dropna()
# =============================================================================
