# -*- coding: utf-8 -*-
"""
@Date: 2024/5/26 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: ods_info_full_asset_base_details.py
@Description: 投资品种明细
"""
import pandas as pd
from datetime import datetime

from __init__ import path
from utils import utils_data

ASSET_TABLE_NAME = 'ods_info_full_asset_base_details'


if __name__ == '__main__':
    # exchange：表示交易所，可以存储交易所的名称，如"北交所"。
    # board_type：表示板块类型，可以存储板块的名称，如"主板"、"创业板"、"科创板"、"新三板"等。
    data = [['stock', '主板', 0.1, ''],
            ['stock', '创业板', 0.2, ''],
            ['stock', '科创板', 0.2, ''],
            ['stock', '新三板', 0.1, ''],
            ['stock', '北交所', 0.3, ''],
            ['stock', 'ST', 0.05, ''],
            ['stock', 'ETF', 0.1, ''],
            ['stock', '指数', 0.1, ''],
            ]
    asset_varietie_df = pd.DataFrame(data, columns=['asset', 'board_type','price_limit_rate','remark'])
    asset_varietie_df['insert_timestamp'] = datetime.now().strftime("%F %T")
    utils_data.output_database(asset_varietie_df, filename=ASSET_TABLE_NAME, if_exists='replace')

