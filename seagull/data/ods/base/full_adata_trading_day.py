# -*- coding: utf-8 -*-
"""
@Date: 2024/5/5 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: full_adata_trading_day.py
@Description: 交易日(ods/base/full_adata_trading_day)
@Update cycle: year
"""
import os
from datetime import datetime 

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_data, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def _full_adata_trading_day_1(year):
    """
    获取指定年份的交易日
    # 2005年开始至今的交易日,不能翻墙
    """
    trading_day_df = adata.stock.info.trade_calendar(year=int(year))
    # trading_day_df = adata.stock.info.trade_calendar(year=2025)
    logger.info(f'trading_year: {year}')
    return trading_day_df


def get_full_adata_trading_day():
    """
    获取所有A股市场的交易日信息
    Returns:

    """
    two_years_later = str(datetime.today().year + 2)
    years = pd.date_range(start='2005', end=two_years_later, freq='YE').year

    trading_day_df = pd.concat([_full_adata_trading_day_1(year) for year in years], ignore_index=True)
    # trading_day_df = years_df.groupby('year').apply(self._ods_adata_trading_day_1)
    return trading_day_df


def pipeline(trading_day_df):
    get_full_adata_trading_day()
    utils_data.output_database(trading_day_df, filename='ods_base_full_adata_trading_day', if_exists='replace')

        
if __name__ == '__main__':
    pipeline()
    