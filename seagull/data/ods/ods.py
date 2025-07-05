# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取ods层基本信息(data_ods)
"""
import os
import argparse
from datetime import datetime, timedelta
from sqlalchemy import String  # Float, Numeric, 

import adata
import efinance as ef
import baostock as bs
import pandas as pd

from seagull.settings import PATH
from seagull.data import (data_loading,
                  ods_part_baostock_index_api,
                  ods_info_incr_baostock_trade_stock_api)
from seagull.utils import utils_database, utils_log, utils_data


log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_freq', type=str, default='1d', help='')

    parser.add_argument('--data_type', type=str, default='证券代码', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()

    ods_data = odsData()
