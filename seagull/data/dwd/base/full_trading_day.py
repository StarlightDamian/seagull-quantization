# -*- coding: utf-8 -*-
"""
@Date: 2025/6/22 22:23
@Author: Damian
@Email: zengyuwei1995@163.com
@File: full_trading_day.py
@Description: 
"""
import os

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_data, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def get_full_trading_day(filename='ods_base_full_adata_trading_day'):
    with utils_database.engine_conn('POSTGRES') as conn:
        trading_day_df = pd.read_sql(filename, con=conn.engine)
    trading_day_df = trading_day_df.rename(columns={'trade_date': 'date',
                                                    'day_week': 'week'})
    return trading_day_df


def pipeline(trading_day_df):
    trading_day_df = get_full_trading_day()
    utils_data.output_database(trading_day_df,
                               filename='dwd_base_full_trading_day', if_exists='replace')


if __name__ == '__main__':
    pipeline()
