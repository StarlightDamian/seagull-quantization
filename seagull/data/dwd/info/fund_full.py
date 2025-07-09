# -*- coding: utf-8 -*-
"""
@Date: 2025/6/24 23:31
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_full.py
@Description: 
"""
import os

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_log, utils_database, utils_character


def get_dwd_info_fund_incr():
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql('ods_info_fund_full_adata', con=conn.engine)

    fund_df = fund_df.rename(columns={'fund_code': 'asset_code',
                                      'short_name': 'code_name',
                                      'net_value': 'prev_close',
                                      })
    fund_df.market_code = fund_df.market_code.map({1: 'sh',
                                                   0: 'sz',
                                                   })
    fund_df['full_code'] = fund_df.market_code + '.' + fund_df.asset_code
    return fund_df
    # utils_data.output_database(fund_df, filename='dwd_info_fund_incr', if_exists='replace')

def pipeline():
    trade_df = dwd_stock_base()
    utils_data.output_database(trade_df, filename='dwd_info_stock_incr_2', if_exists='replace')
    # trade_df.to_csv(f'{PATH}/_file/dwd_info_stock_incr.csv', index=False)

if __name__ == '__main__':
    pipeline()
