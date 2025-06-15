# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 01:07:54 2024

@author: awei
"""

import vectorbt as vbt
import efinance as ef
import pandas as pd

# import numpy as np
from __init__ import path
from utils import utils_database
# ETF dictionary
etf_dict = {
    '510300': '沪深300ETF',
    '159681': '创50ETF',
    '512000': '券商ETF',
    '512880': '证券ETF',
    '515000': '科技ETF',
    '512480': '半导体ETF',
    '512760': '芯片ETF',
    '510150': '消费ETF',
    '518880': '黄金ETF',
    '159509': '纳指科技ETF',
    '513500': '标普500ETF',
    '513600': '恒生指数ETF',
    '159869': '游戏ETF',
    '516390': '新能源汽车ETF',
}

# Time frequencies
freq_dict = {
    5: '5min',
    15: '15min',
    30: '30min',
    60: '60min',
    #101: '日线',
    #102: '周线',
    #103: '月线',
}

def get_rsi(data, window=14):
    rsi = vbt.RSI.run(data, window=window).rsi.fillna(50)
    return rsi


def dataset(klt):
    etf_freq_dict = ef.stock.get_quote_history(list(etf_dict.keys()),
                                               klt=str(klt),
                                               beg='20240101',
                                               )
    # etf_freq_df = pd.concat(etf_freq_dict, names=['code', 'row']).reset_index(level='row', drop=True)
    etf_freq_df = pd.concat({k: v for k, v in etf_freq_dict.items()}).reset_index(drop=True)
    etf_freq_df = etf_freq_df.rename(columns={'日期': 'time',
                                              '股票代码': 'full_code',
                                              '收盘': 'close'})
    etf_freq_df['time'] = pd.to_datetime(etf_freq_df['time'])
    etf_freq_df = etf_freq_df.pivot(index='time', columns='full_code', values='close')
    # etf_freq_df.columns = [etf_dict[code] for code in etf_freq_df.columns]
    return etf_freq_df

def strategy(subtable_df, data):
    window_fast, window_slow, window_signal = subtable_df[['window_fast', 'window_slow', 'window_signal']].values[0]

    # https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/indicators/nb.py#L171
    macd = vbt.MACD.run(
        close=data,  # close: 2D数组，表示收盘价
        fast_window=window_fast,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
        slow_window=window_slow,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
        signal_window=window_signal,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
        macd_ewm=False,  # #布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
        signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
        adjust=False #布尔值，是否在计算EMA时进行调整
        #cache_dict,字典，用于缓存计算结果
    )
    
    # 计算 DIF 和 DEA 的斜率
    dif_slope = macd.macd.diff()  # DIF 线的斜率
    dea_slope = macd.signal.diff()  # DEA 线的斜率
    
    # 交易信号
    #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
    #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
    # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
    entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0) & (dea_slope.shift(1) < 0)
    
    # 卖出信号：DIF 的斜率从正变负
    exits = (dif_slope < 0) & (dif_slope.shift(1) > 0)
    
    global entries1,exits1
    entries1=entries
    exits1=exits
    entries_exits_t = pd.concat([entries, exits],
                                axis=1,
                                keys=['entries', 'exits'],
                                names=['signal_type', 'macd_fast_window', 'macd_slow_window', 'macd_signal_window', 'macd_macd_ewm', 'macd_signal_ewm', 'full_code']
                                ).T
    return entries_exits_t

def pipeline(subtable):
    klt = subtable.name
    
    etf_data = dataset(klt)
    entries_exits_t = strategy(subtable, etf_data)
    # entries_exits.columns = entries_exits.columns.droplevel(0)# 不需要去除第一行column
    entries_exits_t = entries_exits_t.T

# =============================================================================
#     macd_entries_1 = entries.iloc[-1]  # Get the last row (most recent data)
#     macd_exits_1 = exits.iloc[-1]
#     
#     
#     
#     # 使用列表推导生成结果列
#     result = [
#         1 if entry else -1 if exit else 0
#         for entry, exit in zip(macd_entries_1, macd_exits_1)
#     ]
# =============================================================================
    
    # 转换为 pandas Series 或 DataFrame（可选）
    #result_series = pd.Series(result)
    #result_series.index = etf_data.columns
    #entries_exits_t.index = entries_exits_t.index.strftime('%Y-%m-%d')
    # Step 1: 按日期分组，保留每一天的最后一条记录
    
    entries_exits_t.index = pd.MultiIndex.from_arrays(
    [entries_exits_t.index.strftime('%F'),  # 日期部分
     entries_exits_t.index.strftime('%F %T')], # 时间部分
    names=['date', 'time'])  # MultiIndex 的名称

    global entries_exits_t1
    entries_exits_t1=entries_exits_t
    last_entries_per_day = entries_exits_t.groupby(entries_exits_t.index.get_level_values('date')).tail(1)

    # Step 2: 将 index 格式修改为 %F %T
    #last_entries_per_day.index = last_entries_per_day.index.strftime('%Y-%m-%d %H:%M:%S')

    return last_entries_per_day.T#.tail(1)

def keep_levels_by_name(df, level_names):
    new_columns = pd.MultiIndex.from_arrays([df.columns.get_level_values(name) for name in level_names])
    new_columns.names = level_names
    return new_columns

if __name__ == '__main__':
# =============================================================================
#     with utils_database.engine_conn('postgre') as conn:
#         dwd_portfolio_base = pd.read_sql("dwd_info_nrtd_portfolio_base", con=conn.engine)
#     dwd_portfolio_base = dwd_portfolio_base[~(dwd_portfolio_base.prev_close=='-')]
#     etf_dict = dict(zip(dwd_portfolio_base['asset_code'], dwd_portfolio_base['code_name']))
# 
# =============================================================================
    strategy_params_list = [{'klt': 5,'window_fast': 5,'window_slow': 13, 'window_signal':5,},#6,13,4
                            {'klt': 15,'window_fast': 5,'window_slow': 13, 'window_signal':5,},#9,16,5
                            {'klt': 30,'window_fast': 5,'window_slow': 13, 'window_signal':5,},
                            {'klt': 60,'window_fast': 8,'window_slow': 21, 'window_signal':7,},
                            {'klt': 101,'window_fast': 12,'window_slow': 26, 'window_signal':9,},
                            {'klt': 102,'window_fast': 12,'window_slow': 26, 'window_signal':9,},
                            {'klt': 103,'window_fast': 12,'window_slow': 26, 'window_signal':9,},
    ]
    strategy_params_pd = pd.DataFrame(strategy_params_list)
    
    
    # Calculate MACD histogram for all ETFs and time frequencies
    #macd_entries_exits_raw_df = strategy_params_pd.groupby('klt').apply(pipeline)
    entries_exits_raw = strategy_params_pd.groupby('klt').apply(pipeline)
    entries_exits = entries_exits_raw.fillna(0)
    # entries_exits.columns.names=['klt', 'signal_type', 'macd_fast_window', 'macd_slow_window', 'macd_signal_window', 'macd_macd_ewm', 'macd_signal_ewm', 'full_code']
    entries_exits=entries_exits.T
    #entries_exits.index = pd.to_datetime(entries_exits.index)

    date = '2024-09-30'
    # 筛选出列名为 'entries' 的数据
    entries_df = entries_exits.loc[entries_exits.index.get_level_values('date') == date, pd.IndexSlice[:, 'entries']]
    entries_df.columns = keep_levels_by_name(entries_df, ['klt', 'full_code'])
    
    exits_df = entries_exits.loc[entries_exits.index.get_level_values('date') == date, pd.IndexSlice[:, 'exits']]
    exits_df.columns = keep_levels_by_name(exits_df, ['klt', 'full_code'])
    #entries_df = entries_df.applymap(lambda x: 1 if x is True else 0)
    #exits_df = exits_df.applymap(lambda x: -1 if x is True else 0)

# =============================================================================
#     result = [
#         1 if entry else -1 if exit else 0
#         for entry, exit in zip(entries_df, exits_df)
#     ]
# =============================================================================
    entries_exits_df = entries_df.astype(int) - exits_df.astype(int)
    
    entries_exits_df.columns
    entries_exits_df.values[0].reshape(7,14)
    
    #((entries_exits.columns.get_level_values('klt') == 101)&(entries_exits.columns.get_level_values('signal_type') == 'exits'))
    #entries_exits.xs(103, level='klt')
    
    #entries_exits_df['sum'] = entries_exits_df.sum(axis=1)
    
    #entries_exits_df.iloc[-1,entries_exits_df.columns.get_level_values('klt') == 5]
    #entries_exits_df.iloc[:,entries_exits_df.columns.get_level_values('klt') == 101]

# =============================================================================
#     macd_entries_exits_df = macd_entries_exits_raw_df
#     macd_entries_exits_df.index = macd_entries_exits_df.index.map(freq_dict)
#     #macd_entries_exits_df = macd_entries_exits_df.sort_index(ascending=True)
#     etf_sum = macd_entries_exits_df.sum()
#     
#     macd_entries_exits_df = macd_entries_exits_df.T
#     macd_entries_exits_df['sum'] = etf_sum
#     macd_entries_exits_df = macd_entries_exits_df.sort_values(by='sum', ascending=False)
#     del macd_entries_exits_df['sum']
#     macd_entries_exits_df.index = macd_entries_exits_df.index.map(etf_dict)
#     macd_entries_exits_df = macd_entries_exits_df.replace({1:'买入', -1:'卖出', 0:''})
#     macd_entries_exits_df.to_csv(f'{path}/_file/macd_entries_exits.csv')
# =============================================================================
    
# =============================================================================
#     # macd
#     macd_results = macd_results*1000
#     macd_results = macd_results.round(3)
#     macd_results.columns = macd_results.columns.get_level_values('code_name')
#     macd_results.columns.name = 'macd能量柱'
#     macd_results.to_csv(f'{path}/_file/etf_freq_macd.csv')
#     # Display the results
#     print(macd_results)
# =============================================================================
    