# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:45:25 2024

@author: awei
投资品种(data_ods_info_full_asset_price_limit)
投资品类表(data_investment_varietie)，很少修改
"""
import pandas as pd
from datetime import datetime

from __init__ import path
from data import data_utils

ASSET_TABLE_NAME = 'ods_info_full_asset_price_limit'


if __name__ == '__main__':
    #exchange：表示交易所，可以存储交易所的名称，如"北交所"。
    #board_type：表示板块类型，可以存储板块的名称，如"主板"、"创业板"、"科创板"、"新三板"等。
    data = [['stock', '主板', 10, ''],
            ['stock', '创业板', 20, ''],
            ['stock', '科创板', 20, ''],
            ['stock', '新三板', 10, ''],
            ['stock', '北交所', 30, ''],
            ['stock', 'ST', 5, ''],
            ['stock', 'ETF', 10, ''],
            ['stock', '指数', 100, ''],
            ]
    asset_varietie_df = pd.DataFrame(data, columns=['asset', 'board_type','price_limit_pct','remark'])
    asset_varietie_df['insert_timestamp'] = datetime.now().strftime("%F %T")
    data_utils.output_database(asset_varietie_df, filename=ASSET_TABLE_NAME, if_exists='replace')
    
# =============================================================================
#     #exchange：表示交易所，可以存储交易所的名称，如"北交所"。
#     #board_type：表示板块类型，可以存储板块的名称，如"主板"、"创业板"、"科创板"、"新三板"等。
#     
#     with base_connect_database.engine_conn('postgre') as conn:
#         all_stock = pd.read_sql('all_stock', con=conn.engine)
#         stock_basic = pd.read_sql('stock_basic', con=conn.engine)
#     stock_basic['type'] = stock_basic['type'].map({'1': '股票',
#                                                    '2': '指数',
#                                                    '3': '其它',
#                                                    '4': '可转债',
#                                                    '5': 'ETF',})
#     stock_type_dict = dict(zip(stock_basic['code'], stock_basic['type']))
#     
#     all_stock[['tpye', 'price_limit']] = ''
#     all_stock.loc[all_stock.code.str.contains('sh.60|sz.00'), ['tpye','price_limit']] = '主板', 0.1
#     all_stock.loc[all_stock.code.str.contains('\.300|sz.301'), ['tpye','price_limit']] = '创业板', 0.2
#     all_stock.loc[all_stock.code.str.contains('\.688|\.689'), ['tpye','price_limit']] = '科创板', 0.2
#     all_stock.loc[all_stock.code.str.contains('\.430|\.830'), ['tpye','price_limit']] = '新三板', 0.1
#     all_stock.loc[all_stock.code.str.contains('bj.'), ['tpye','price_limit']] = '北交所', 0.3
#     
#     all_stock.loc[all_stock.tpye=='', 'tpye'] =     all_stock.loc[all_stock.tpye=='', 'code'].map(stock_type_dict)
#     
#     all_stock.loc[all_stock.tpye=='指数', 'price_limit'] = 10
#     #all_stock.loc[all_stock.code_name.str.contains('指数'), ['tpye','price_limit']] = '指数', 10
#     #all_stock.to_csv(f'{path}/data/all_stock_copy.csv', index=False)
#     
#     with base_connect_database.engine_conn('postgre') as conn:
#         all_stock.to_sql(STOCK_COPY_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')
#         eval_df.to_sql(TABLE_NAME, con=conn.engine, index=False, if_exists='replace')
# =============================================================================
