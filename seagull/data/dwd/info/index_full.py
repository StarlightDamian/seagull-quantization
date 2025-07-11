# -*- coding: utf-8 -*-
"""
@Date: 2025/7/10 19:36
@Author: Damian
@Email: zengyuwei1995@163.com
@File: index_full.py
@Description: 指数
"""
import os
import pandas as pd
from seagull.settings import PATH
from seagull.utils import utils_data, utils_database, utils_log, utils_character
from seagull.utils.api import utils_api_baostock


def dwd_info_stock_baostock():
    # ods_info_stock_incr_baostock.columns = ['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status',
    # 'insert_timestamp']
    with utils_database.engine_conn("POSTGRES") as conn:
        baostock_df = pd.read_sql("select * from ods_info_stock_incr_baostock where type='2'", con=conn.engine)

    baostock_df = utils_api_baostock.split_baostock_code(baostock_df)

    baostock_df = baostock_df.rename(columns={'status': 'trade_status',
                                              'ipoDate': 'listing_date',
                                              'outDate': 'delisting_date',
                                              })
    return baostock_df


def pipeline():
    fund_df = dwd_info_stock_baostock()
    fund_df['settlement_cycle'] = 1  # A股默认T+1
    utils_data.output_database(fund_df, filename='dwd_info_index_full', if_exists='replace')


if __name__ == '__main__':
    pipeline()
