# -*- coding: utf-8 -*-
"""
@Date: 2025/7/11 9:52
@Author: Damian
@Email: zengyuwei1995@163.com
@File: futures_full.py
@Description: 期货
"""
import efinance as ef
import pandas as pd
from seagull.utils import utils_data, utils_database


def dwd_info_futures_efinance():
    futures_df = ef.futures.get_futures_base_info()
    futures_df = futures_df.rename(columns={"期货代码": "asset_code",
                                            "行情ID": "full_code",
                                            "期货名称": "code_name",
                                            "市场类型": "market"})
    futures_df['settlement_cycle'] = 0  # 期货默认T+0
    futures_df = futures_df[['full_code', 'asset_code', 'code_name', 'market', 'settlement_cycle']]
    return futures_df


def pipeline():
    futures_df = dwd_info_futures_efinance()
    utils_data.output_database(futures_df, filename='dwd_info_futures_full', if_exists='replace')


if __name__ == '__main__':
    pipeline()
