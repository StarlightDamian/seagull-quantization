# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:11:31 2025

@author: awei
(macd)
"""

import os
import argparse

import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt

from __init__ import path
from utils import utils_log, utils_character, utils_database

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

def vbt_macd(data):
    macd = vbt.MACD.run(
        close=data,  # close: 2D数组，表示收盘价
        fast_window=12,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
        slow_window=26,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
        signal_window=9,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
        macd_ewm=False, # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
        signal_ewm=True,  # 布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
        adjust=False,  # 布尔值，是否在计算EMA时进行调整
        # cache_dict,  # 字典，用于缓存计算结果
        )
    return macd
    
def dataset():
    symbols = ['AAPL', 'MSFT']
    data = vbt.YFData.download(symbols, start='2021-01-01', end='2022-01-01', missing_index='drop').get('Close')
    data.columns = pd.MultiIndex.from_product([data.columns, ['close']], names=['symbol', 'parms'])
    return data
    
#def pipeline(data):
#    macd = vbt_macd(data)
#    return macd

def freq2vbt(df, columns=['close']):
    df.index = df.date
    df = df[columns]
    df.columns.name = 'parms'
    return df.T

def vbt2freq(df_t):
    df_t = df_t.droplevel('full_code')
    return df_t.T

def macd2df(macd, task_type=None, freq='d', adj_type='pre'):
    """
    dif_line = macd.macd  # 提取 DIF 线（即 MACD 线）
    dif_slope = np.diff(dif_line)   # 计算 DIF 线的斜率（一阶导数）
    dif_acceleration = np.diff(dif_slope)  # 计算 DIF 线的二阶导数（加速度）
    """
    if task_type=='slope':  # dif斜率
        df = macd.macd.diff().T.groupby(level='full_code').apply(vbt2freq)
    elif task_type=='acceleration':  # diff2,dif线的二阶导
        df = macd.macd.diff().diff().T.groupby(level='full_code').apply(vbt2freq)
    elif task_type=='hist':  # 能量柱
        df = macd.hist.T.groupby(level='full_code').apply(vbt2freq)
    # print('df.columns',df.columns)
    # print('df.index',df.index)
    df.columns = [f"{parms}_{task_type}_{fast}_{slow}_{signal}" 
    for fast, slow, signal, _, _, parms in df.columns]
    df = df.reset_index()
    
    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    df['freq'] = freq
    df['adj_type'] = adj_type   # adjustment_type as adj_type in ['None', 'pre', 'post']
    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
    
    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    df['primary_key'] = (df['time'].astype(str) +
                         df['full_code'].astype(str) +
                         df['freq'].astype(str) +
                         df['adj_type'].astype(str)
                        ).apply(utils_character.md5_str) # md5（时间、带后缀代码、频率、复权类型）
    return df
    
def diff(df, numeric_columns):
    # 计算前1日、前5日、前22日的差值
    for numeric in numeric_columns:
        df[f'{numeric}_diff1'] = df[numeric] - df[numeric].shift(1)
        df[f'{numeric}_diff5'] = df[numeric] - df[numeric].shift(5)
        df[f'{numeric}_diff30'] = df[numeric] - df[numeric].shift(22)
    return df

def diff_hist(df, numeric_columns):
    for numeric in numeric_columns:
        df[f'{numeric}_hist_diff1'] = df[f'{numeric}_hist_12_26_9'] - df[f'{numeric}_hist_12_26_9'].shift(1)
    return df

def pipeline(df, columns, numeric_columns, freq='d', adj_type='pre'):
    data_t = df.groupby('full_code').apply(freq2vbt, columns=columns)#, include_groups=False
    data = data_t.T
    macd = vbt_macd(data)
    slope_df = macd2df(macd, task_type='slope', freq=freq, adj_type=adj_type)
    acceleration_df = macd2df(macd, task_type='acceleration', freq=freq, adj_type=adj_type)
    hist_df = macd2df(macd, task_type='hist', freq=freq, adj_type=adj_type)
    
    slope_df = slope_df[['primary_key'] + [f'{column}_slope_12_26_9' for column in columns]]
    acceleration_df = acceleration_df[['primary_key'] + [f'{column}_acceleration_12_26_9' for column in columns]]
    hist_df = hist_df[['primary_key'] + [f'{column}_hist_12_26_9' for column in columns]]
    
    df = pd.merge(df, slope_df, on='primary_key', how='left')
    df = pd.merge(df, acceleration_df, on='primary_key', how='left')
    df = pd.merge(df, hist_df, on='primary_key', how='left')
    
    df = diff(df, numeric_columns)
    df = diff_hist(df, numeric_columns)
    return df

def trading_signal(df):
    df.index = df['date']
    macd = vbt_macd(df.close)
    
    # macd_series = macd.macd
    # macd_df = macd_series[pd.notnull(macd_series)] # 第一行非空
    
    # 计算 DIF 和 DEA 的斜率
    dif_slope = macd.macd.diff()  # DIF 线的斜率
    dea_slope = macd.signal.diff()  # DEA 线的斜率
    
    # 交易信号
    #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
    #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
    # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
    entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0)  # & (dea_slope.shift(1) < 0)
    
    # 卖出信号：DIF 的斜率从正变负
    exits = (dif_slope < 0) & (dif_slope.shift(1) > 0)
    
    entries_exits_df = pd.concat([entries, exits], axis=1, keys=['entries', 'exits'])
    return entries_exits_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-06-01', help='End time for backtesting')
    args = parser.parse_args()
    
    with utils_database.engine_conn('postgre') as conn:
        stock_daily_df = pd.read_sql(f"""
            select 
                full_code,
                date,
                close,
                primary_key
            from
                dwd_ohlc_incr_stock_daily
            where
                date between '{args.date_start}' and '{args.date_end}'
                and full_code='300059.sz'
                --and full_code='510300.sh'
                """, con=conn.engine)
    
    entries_exits_df = trading_signal(stock_daily_df)
    stock_daily_series = stock_daily_df[['date', 'close']].set_index('date')#['close']
    
    merged_df = stock_daily_series.join(entries_exits_df, how='left')
    
    # 选择日期范围
    date_start_mini = '2024-08-01'
    date_end_mini = '2025-01-01'
    selected_data = merged_df.loc[(merged_df.index >= date_start_mini) & (merged_df.index <= date_end_mini)]

    
    # 创建图形
    plt.figure(figsize=(14, 7))
    
    # 绘制收盘价
    plt.plot(selected_data.index, selected_data['close'], label='Close Price', color='blue')
    
    # 标记买入位置 (entries == True)
    buy_points = selected_data[selected_data['entries'] == True]
    plt.scatter(buy_points.index, buy_points['close'], color='green', marker='^', label='Buy', s=100)
    
    # 标记卖出位置 (exits == True)
    sell_points = selected_data[selected_data['exits'] == True]
    plt.scatter(sell_points.index, sell_points['close'], color='red', marker='v', label='Sell', s=100)
    
    # 添加标题和标签
    plt.title(f"Stock Price and Buy/Sell Points ({date_start_mini} to {date_end_mini})")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # 格式化日期显示
    plt.xticks(rotation=45)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    