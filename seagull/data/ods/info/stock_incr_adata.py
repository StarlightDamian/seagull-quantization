# -*- coding: utf-8 -*-
"""
@Date: 2024/10/28 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_adata.py
@Description: (ods/info/stock_incr_adata)
@Update cycle: day
"""
import os

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_log, utils_database, utils_character

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def dwd_info_incr_adata_stock_base():
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_base_df = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
    stock_base_df = stock_base_df[['stock_code', 'exchange']]
    stock_base_df = stock_base_df.rename(columns={'exchange': 'market_code',
                                                  'stock_code': 'asset_code',
                                                  })
    
    #capital_flow_df = pd.merge(capital_flow_df, stock_base_df, on='asset_code')
    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    
    stock_base_df['market_code'] = stock_base_df['market_code'].str.lower()
    stock_base_df['full_code'] = stock_base_df['asset_code'] + '.' + stock_base_df['market_code']
    return stock_base_df


def associated_primary_key(df):
    """
    df必备字段 df.columns = ['date', 'freq', 'adj_type']

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    stock_base_df = dwd_info_incr_adata_stock_base()
    stock_base_df = stock_base_df[['full_code', 'asset_code', 'market_code']]
    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    df = pd.merge(df, stock_base_df, on='asset_code', how='left')
    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    df['primary_key'] = (df['time'].astype(str) +
                         df['full_code'].astype(str) +
                         df['freq'].astype(str) +
                         df['adj_type'].astype(str)
                         ).apply(utils_character.md5_str) # md5（时间、带后缀代码、频率、复权类型）
    return df  #[['primary_key', 'full_code', 'asset_code', 'market_code', 'date', 'time', 'freq', 'adj_type']]

    
def dwd_pipeline():
    stock_base_df = dwd_info_incr_adata_stock_base()
    utils_data.output_database(stock_base_df,
                               filename='dwd_info_incr_adata_stock_base',
                               if_exists='replace')


def ods_info_incr_adata_stock_base():
    ods_adata_stock_base = adata.stock.info.all_code()
    print(ods_adata_stock_base)  # ['stock_code', 'short_name', 'exchange', 'list_date']
    utils_data.output_database(ods_adata_stock_base,
                               filename='ods_info_incr_adata_stock_base',
                               if_exists='replace')


if __name__ == '__main__':
    ods_info_incr_adata_stock_base()
    dwd_pipeline()
