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
# ETF dictionary
etf_dict = {
    '510300': '沪深300ETF',
    '159681': '创50ETF',
    '512000': '券商ETF',
    '512880': '证券ETF',
    '515000': '科技ETF',
    '512480': '半导体ETF',
    '512760': '芯片ETF',
}

# Time frequencies
freq_dict = {
    '5': '5min',
    '15': '15min',
    '30': '30min',
    '60': '60min',
    '101': '日线',
    '102': '周线',
    '103': '月线',
}

def get_macd_histogram(data, fast_window=12, slow_window=26, signal_window=9):
    macd = vbt.MACD.run(
        close=data,
        fast_window=fast_window,
        slow_window=slow_window,
        signal_window=signal_window,
        macd_ewm=True,
        signal_ewm=True,
        adjust=False
    )
    return macd.hist

def get_rsi(data, window=14):
    rsi = vbt.RSI.run(data, window=window).rsi.fillna(50)
    return rsi

def process_etf_data(klt):
    etf_freq_dict = ef.stock.get_quote_history(list(etf_dict.keys()), klt=klt)
    # etf_freq_df = pd.concat(etf_freq_dict, names=['code', 'row']).reset_index(level='row', drop=True)
    etf_freq_df = pd.concat({k: v for k, v in etf_freq_dict.items()}).reset_index(drop=True)
    etf_freq_df = etf_freq_df.rename(columns={'日期': 'date',
                                                      '股票名称': 'code_name',
                                                      '收盘': 'close'})
    etf_freq_df['date'] = pd.to_datetime(etf_freq_df['date'])
    etf_freq_df = etf_freq_df.pivot(index='date', columns='code_name', values='close')
    # etf_freq_df.columns = [etf_dict[code] for code in etf_freq_df.columns]
    return etf_freq_df


def calculate_indicators_for_all_etfs(freq_dict):
    macd_result_dict = {}
    rsi_result_dict = {}
    for (freq, freq_name) in freq_dict.items():
        etf_data = process_etf_data(freq)
        macd_hist = get_macd_histogram(etf_data)
        rsi = get_rsi(etf_data)
        macd_result_dict[freq_name] = macd_hist.iloc[-1]  # Get the last row (most recent data)
        rsi_result_dict[freq_name] = rsi.iloc[-1]  # Get the last row (most recent data)
    return pd.DataFrame(macd_result_dict).T, pd.DataFrame(rsi_result_dict).T


if __name__ == '__main__':
    strategy_params_list = [{'klt': '5','window_fast': 5,'window_slow': 13,'window_signal':5,},
                            {'klt': '15','window_fast': 5,'window_slow': 13,'window_signal':5,},
                            {'klt': '30','window_fast': 5,'window_slow': 13,'window_signal':5,},
                            {'klt': '60','window_fast': 8,'window_slow': 21,'window_signal':7,},
                            {'klt': '101','window_fast': 12,'window_slow': 26,'window_signal':9,},
                            {'klt': '102','window_fast': 12,'window_slow': 26,'window_signal':9,},
                            {'klt': '103','window_fast': 12,'window_slow': 26,'window_signal':9,},
    ]
    
    # Calculate MACD histogram for all ETFs and time frequencies
    macd_results, rsi_results = calculate_indicators_for_all_etfs(freq_dict)
    
    # macd
    macd_results = macd_results*1000
    macd_results = macd_results.round(3)
    macd_results.columns = macd_results.columns.get_level_values('code_name')
    macd_results.columns.name = 'macd能量柱'
    macd_results.to_csv(f'{path}/_file/etf_freq_macd.csv')
    # Display the results
    print(macd_results)
    
    # rsi
    rsi_results = rsi_results.round(3)
    rsi_results.columns = rsi_results.columns.get_level_values('code_name')
    rsi_results.columns.name = 'rsi'
    rsi_results.to_csv(f'{path}/_file/etf_freq_rsi.csv')