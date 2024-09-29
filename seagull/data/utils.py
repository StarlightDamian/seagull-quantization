# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 03:08:26 2024

@author: awei
本项目数据通用处理工具包(data_utils)
"""
import os

import pandas as pd

from __init__ import path
from base import base_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')

        
def split_baostock_code(df):
    # 去掉前面的“sh.”、“sz.”、“bj.”
    # data_raw_df['asset_code'] = data_raw_df['asset_code'].str.replace(r'^[a-z]{2}\.', '', regex=True)
    
    # 提取代码部分
    df['market_code'] = df['asset_code'].str.split('.').str[0].str.upper()
    df['asset_code'] = df['asset_code'].str.split('.').str[1]
    
    # 关联数据
    # with base_connect_database.engine_conn('postgre') as conn:
    #     ods_investment_suffix = pd.read_sql('ods_investment_suffix', con=conn.engine)
    # result = pd.merge(df, ods_investment_suffix[['ticker_suffix']], how='left', on='ticker_suffix')
    
    # 输出结果
    return df#result[['asset_code', 'trade_status', 'ticker_suffix']]

def re_get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result


if __name__ == '__main__':
    df = pd.DataFrame([['bj.430017', 1],['sh.430017', 1],['sz.002906',1]],columns=['code','trade_status'])
    result_df = split_baostock_code(df)
    
