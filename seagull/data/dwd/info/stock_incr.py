# -*- coding: utf-8 -*-
"""
@Date: 2025/6/22 14:47
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr.py
@Description: 上市公司基本信息大宽表(dwd/info/stock_incr)
"""
import os

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_log, utils_database, utils_character


def dwd_info_stock_incr():
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_base_df = pd.read_sql("ods_info_stock_incr_adata", con=conn.engine)

    stock_base_df = stock_base_df[['stock_code', 'exchange']]
    stock_base_df = stock_base_df.rename(columns={'exchange': 'market_code',
                                                  'stock_code': 'asset_code',
                                                  })

    stock_base_df['market_code'] = stock_base_df['market_code'].str.lower()
    stock_base_df['full_code'] = stock_base_df['asset_code'] + '.' + stock_base_df['market_code']
    return stock_base_df


def pipeline():
    stock_base_df = dwd_info_stock_incr()
    utils_data.output_database(stock_base_df,
                               filename='dwd_info_stock_incr',
                               if_exists='replace'
                               )


if __name__ == '__main__':
    pipeline()
