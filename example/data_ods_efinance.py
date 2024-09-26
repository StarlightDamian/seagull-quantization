# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:58:10 2024

@author: awei
(data_api_efinance)
"""

import efinance as ef
import pandas as pd
stock_code = '430017'
df = ef.stock.get_quote_history(stock_code)
def realtime(stock_code):
    realtime_df = ef.stock.get_realtime_quotes()
    #realtime_df.columns=['股票代码', '股票名称', '涨跌幅', '最新价', '最高', '最低', '今开', '涨跌额', '换手率', '量比',
     #      '动态市盈率', '成交量', '成交额', '昨日收盘', '总市值', '流通市值', '行情ID', '市场类型', '更新时间',
    #       '最新交易日']
    realtime_df = realtime_df.loc[realtime_df.股票代码.isin(stock_code), ['股票代码', '涨跌幅', '最新价', '最高', '最低', '今开', '涨跌额', '换手率', '成交量', '成交额']]
    return realtime_df

def last(stock_code):
    last_dict = ef.stock.get_quote_history(stock_code)
    last_df = pd.concat({k: v for k, v in last_dict.items()})
    #last_df = last_df.reset_index(level=0).rename(columns={'level_0': '股票代码'})
    last_df = last_df.drop_duplicates('股票代码',keep='last')
    last_df = last_df[['股票代码', '涨跌幅', '最高', '最低', '涨跌额', '换手率', '成交量', '成交额']]
    return last_df