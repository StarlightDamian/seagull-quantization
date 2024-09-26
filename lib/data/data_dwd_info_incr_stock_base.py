# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:06:45 2024

@author: awei
上市公司的附加信息(data_dwd_info_incr_stock_base)
更新周期：7天
问题记录
1.ST可能带帽和摘帽，怎么切换
2.股票发行日期。新股，次新股
3.北交所是交易所，概念和‘主板’，‘创业板’其实不一样
4.缺少ETF
"""
import pandas as pd

from __init__ import path
from base import base_connect_database
from data import data_utils 

def dwd_stock_base(ASSET_TABLE_NAME='ods_info_full_asset_price_limit', DWD_STOCK_BASE_TABLE_NAME='dwd_info_incr_stock_base'):
    with base_connect_database.engine_conn('postgre') as conn:
        date_max = data_utils.maximum_date('ods_info_incr_baostock_trade_stock')
        trade_stock_df = pd.read_sql(f'SELECT * FROM ods_info_incr_baostock_trade_stock WHERE date={date_max}', con=conn.engine)
        stock_base_df = pd.read_sql('ods_info_incr_baostock_stock_base', con=conn.engine)
        investment_varietie_df = pd.read_sql(ASSET_TABLE_NAME, con=conn.engine)
    stock_base_df['board_type'] = stock_base_df['type'].map({'1': '股票',
                                                         '2': '指数',
                                                         '3': '其它',
                                                         '4': '可转债',
                                                         '5': 'ETF',})
    stock_board_dict = dict(zip(stock_base_df['code'], stock_base_df['board_type']))
    investment_varietie_dict = dict(zip(investment_varietie_df['board_type'], investment_varietie_df['price_limit_pct']))
    trade_stock_df[['board_type', 'price_limit_pct']] = ''
    trade_stock_df.loc[trade_stock_df.code.str.contains('sh.60|sz.00'), 'board_type'] = '主板'
    trade_stock_df.loc[trade_stock_df.code.str.contains('\.300|sz.301'),'board_type'] = '创业板'
    trade_stock_df.loc[trade_stock_df.code.str.contains('\.688|\.689'), 'board_type'] = '科创板'
    trade_stock_df.loc[trade_stock_df.code.str.contains('\.430|\.830'), 'board_type'] = '新三板'
    trade_stock_df.loc[trade_stock_df.code.str.contains('bj.'), 'board_type'] = '北交所'
    
    # ST的中文名称会随时间段变化(也存在没有对应中文名的情况)，通过日线的isST字段来判断训练，
    # trade_stock_df.loc[trade_stock_df.code_name.str.contains('ST'), 'board_type'] = 'ST'
    
    trade_stock_df.loc[trade_stock_df.board_type=='', 'board_type'] = trade_stock_df.loc[trade_stock_df.board_type=='', 'code'].map(stock_board_dict)
    
    trade_stock_df['price_limit_pct'] = trade_stock_df['board_type'].map(investment_varietie_dict)
    trade_stock_df = trade_stock_df.rename(columns={"tradeStatus": "trade_status"})
    trade_stock_df = data_utils.split_baostock_code(trade_stock_df)
    trade_stock_df['code_with_ticker_suffix'] = trade_stock_df['ticker_suffix'] + '.' + trade_stock_df['code']  # code_with_ticker_suffix是唯一的
    
    # 结算周期
    trade_stock_df['settlement_cycle'] = 1 # A股默认T+1
    
    trade_stock_df = trade_stock_df[['ticker_suffix','code', 'code_with_ticker_suffix','code_name','board_type','price_limit_pct','trade_status','insert_timestamp']]
    trade_stock_df.to_csv(f'{path}/data/{DWD_STOCK_BASE_TABLE_NAME}.csv', index=False)
    with base_connect_database.engine_conn('postgre') as conn:
        trade_stock_df.to_sql(DWD_STOCK_BASE_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')

if __name__ == '__main__':
    dwd_stock_base()