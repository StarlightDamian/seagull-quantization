# -*- coding: utf-8 -*-
"""
@Date: 2024/8/22 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: full_asset.py
@Description:(ods/base/full_asset)
@Update cycle: permanent
"""
import pandas as pd

from seagull.utils import utils_data
from seagull.settings import PATH

ASSET_TABLE_NAME = 'ods_base_full_asset'

if __name__ == '__main__':
    asset_df = pd.read_csv(f"{PATH}/data/ods_base_full_asset.csv", encoding="gb18030")
    utils_data.output_database(asset_df, filename=ASSET_TABLE_NAME, if_exists='replace')
