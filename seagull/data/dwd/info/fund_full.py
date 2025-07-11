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


def get_dwd_info_fund_full():
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql('ods_info_fund_full_adata', con=conn.engine)

    fund_df = fund_df.rename(columns={'fund_code': 'asset_code',
                                      'short_name': 'code_name',
                                      'net_value': 'prev_close',
                                      })
    fund_df.market_code = fund_df.market_code.map({1: 'sh',
                                                   0: 'sz',
                                                   })
    fund_df['full_code'] = fund_df.asset_code + '.' + fund_df.market_code

    fund_df['price_limit_rate'] = 0.1
    fund_df.loc[fund_df['code_name'].str.contains('科创|创业|双创'), 'price_limit_rate'] = 0.2

    # 结算周期, A股默认T+1
    fund_df['settlement_cycle'] = 1

    fund_df = fund_df[['full_code', 'asset_code', 'market_code', 'code_name', 'prev_close', 'price_limit_rate',
                       'settlement_cycle']]

    return fund_df


def pipeline():
    fund_df = get_dwd_info_fund_full()
    utils_data.output_database(fund_df, filename='dwd_info_fund_full', if_exists='replace')


if __name__ == '__main__':
    pipeline()
