# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:19:56 2024

@author: awei

#macd.macd #macd.macd 返回的是 vbt.MACD.run() 计算结果中的 MACD 线（即 DIF 线）
#dif_slope = macd.macd.diff()  # DIF 线的斜率
#macd_hist = macd.hist # macd能量柱

#idx = pd.IndexSlice
#macd.hist.loc[:, idx[:, :, :, :, :, '000001.sh', :]]
"""
import os

#import numpy as np
import pandas as pd

import vectorbt as vbt
from __init__ import path
from utils import utils_database, utils_log, utils_character, utils_math  #, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

def vbt_macd(data):
    macd = vbt.MACD.run(
        close=data,  # close: 2D数组，表示收盘价
        fast_window=12,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
        slow_window=26,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
        signal_window=9,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
        macd_ewm=False, # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
        signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
        adjust=False, #布尔值，是否在计算EMA时进行调整
        #cache_dict,字典，用于缓存计算结果
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

def macd2df(macd, task_type=None):
    if task_type=='slope':  # 斜率
        df = macd.macd.diff().T.groupby(level='full_code').apply(vbt2freq)
    elif task_type=='hist':  # 能量柱
        df = macd.hist.T.groupby(level='full_code').apply(vbt2freq)
    df.columns = [f"{parms}_{task_type}_{fast}_{slow}_{signal}" 
    for fast, slow, signal, _, _, parms in df.columns]
    df = df.reset_index()
    
    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    df['freq'] = 'd'
    df['adj_type'] = None   # adjustment_type as adj_type in ['None', 'Pre', 'Post']
    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
    
    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    df['primary_key'] = (df['time'].astype(str) +
                         df['full_code'].astype(str) +
                         df['freq'].astype(str) +
                         df['adj_type'].astype(str)
                        ).apply(utils_character.md5_str) # md5（时间、带后缀代码、频率、复权类型）
  
    return df

def diff(df):
    # 计算前1日、前5日、前22日的差值
    df['volume_diff_1'] = df['volume'] - df['volume'].shift(1)
    df['volume_diff_5'] = df['volume'] - df['volume'].shift(5)
    df['volume_diff_30'] = df['volume'] - df['volume'].shift(22)
    
    df['value_traded_diff_1'] = df['value_traded'] - df['value_traded'].shift(1)
    df['value_traded_diff_5'] = df['value_traded'] - df['value_traded'].shift(5)
    df['value_traded_diff_30'] = df['value_traded'] - df['value_traded'].shift(22)
    return df

def diff_hist(df):
    df['value_traded_hist_diff_1'] = df['value_traded_hist_12_26_9'] - df['value_traded_hist_12_26_9'].shift(1)
    df['volume_hist_diff_1'] = df['volume_hist_12_26_9'] - df['volume_hist_12_26_9'].shift(1)
    return df

if __name__ == '__main__':
    full_code_tuple = ('399101.sz', '399102.sz', '000300.sh', '000001.sh', '399106.sz')
    with utils_database.engine_conn('postgre') as conn:
        stock_daily_df = pd.read_sql(f"""
            select
                full_code,
                date,
                volume,
                value_traded,
                turnover,
                primary_key
            from
                dwd_freq_incr_stock_daily
            where
                full_code in {full_code_tuple}""",
                con=conn.engine)
    
    stock_daily_df = stock_daily_df.drop_duplicates('primary_key', keep='first')
    data_t = stock_daily_df.groupby('full_code').apply(freq2vbt, columns=['volume', 'value_traded', 'turnover'], include_groups=False)
    data = data_t.T
    macd = vbt_macd(data)
    slope_df = macd2df(macd, task_type='slope')
    hist_df = macd2df(macd, task_type='hist')
    
    slope_df = slope_df[['primary_key', 'volume_slope_12_26_9', 'value_traded_slope_12_26_9', 'turnover_slope_12_26_9']]
    hist_df = hist_df[['primary_key', 'volume_hist_12_26_9', 'value_traded_hist_12_26_9', 'turnover_hist_12_26_9']]
    
    stock_daily_df = pd.merge(stock_daily_df, slope_df, on='primary_key')
    stock_daily_df = pd.merge(stock_daily_df, hist_df, on='primary_key')
    
    stock_daily_df = diff(stock_daily_df)
    stock_daily_df = diff_hist(stock_daily_df)
    
    # log e 更容易学习特征
    columns_to_transform = ['volume_slope_12_26_9', 'value_traded_slope_12_26_9', 'turnover_slope_12_26_9',
               'volume_hist_12_26_9', 'value_traded_hist_12_26_9', 'turnover_hist_12_26_9',
               'volume_diff_1', 'volume_diff_5', 'volume_diff_30',
               'value_traded_diff_1', 'value_traded_diff_5', 'value_traded_diff_30',
               'value_traded_hist_diff_1', 'volume_hist_diff_1']
    stock_daily_df[columns_to_transform] = stock_daily_df[columns_to_transform].map(utils_math.log_e)
    
