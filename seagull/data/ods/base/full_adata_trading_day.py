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


class TradingDay:
    def __init__(self):
        ...

    @staticmethod
    def _ods_adata_trading_day_1(self, year):
        """
        获取指定年份的交易日
        # 2005年开始至今的交易日,不能翻墙
        """
        trading_day_df = adata.stock.info.trade_calendar(year=int(year))
        # trading_day_df = adata.stock.info.trade_calendar(year=2025)
        logger.info(f'trading_year: {year}')
        return trading_day_df

    def _ods_adata_trading_day(self):
        """
        获取所有A股市场的交易日信息
        Returns:

        """
        two_years_later = str(datetime.today().year + 2)
        years = pd.date_range(start='2005', end=two_years_later, freq='YE').year
        trading_day_df = pd.concat([self._ods_adata_trading_day_1(year) for year in years], ignore_index=True)
        # trading_day_df = years_df.groupby('year').apply(self._ods_adata_trading_day_1)
        utils_data.output_database(trading_day_df, filename='ods_base_full_adata_trading_day', if_exists='replace')

        return trading_day_df

    @staticmethod
    def _dwd_trading_day(self, filename='ods_base_full_adata_trading_day'):
        with utils_database.engine_conn('POSTGRES') as conn:
            trading_day_df = pd.read_sql(filename, con=conn.engine)
        trading_day_df = trading_day_df.rename(columns={'trade_date': 'date',
                                                        'day_week': 'week'})
        utils_data.output_database(trading_day_df, filename='dwd_base_full_trading_day', if_exists='replace')

    def pipeline(self):
        # 更新周期: 年或月
        self._ods_adata_trading_day()
        self._dwd_trading_day()
        
        
if __name__ == '__main__':
    trading_day = TradingDay()
    trading_day.pipeline()
    