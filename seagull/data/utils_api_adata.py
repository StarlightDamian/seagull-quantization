# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:52:59 2024

@author: awei
adata的api数据处理工具(utils_api_adata)
"""

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_character


def full_code_adata(stock_df):
    with utils_database.engine_conn("POSTGRES") as conn:
        df = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
    df = df.rename(columns={'stock_code': 'asset_code',
                            'exchange': 'market_code'})
    df.market_code = df.market_code.str.lower()
    df['full_code'] = df['asset_code'] + '.' + df['market_code']
    stock_df = pd.merge(stock_df, df[['asset_code', 'full_code']], on='asset_code', how='left')
    return stock_df


def primary_key_adata(df):
    """
    df必备字段 df.columns = ['date', 'freq', 'adj_type']
    Args:
        df:

    Returns:

    """
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_base_df = pd.read_sql("dwd_info_stock_incr_adata", con=conn.engine)
    stock_base_df = stock_base_df[['full_code', 'asset_code', 'market_code']]

    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    df = pd.merge(df, stock_base_df, on='asset_code', how='left')
    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')

    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    df['primary_key'] = (df['time'].astype(str) +
                         df['full_code'].astype(str) +
                         df['freq'].astype(str) +
                         df['adj_type'].astype(str)
                         ).apply(utils_character.md5_str)  # md5（时间、带后缀代码、频率、复权类型）
    return df  # [['primary_key', 'full_code', 'asset_code', 'market_code', 'date', 'time', 'freq', 'adj_type']]


if __name__ == '__main__': 
    with utils_database.engine_conn("POSTGRES") as pg_conn:
        adata_stock_label_df = pd.read_sql("ods_flag_full_adata_stock_label", con=pg_conn.engine)
    result_df = full_code_adata(adata_stock_label_df)
    print(result_df)
