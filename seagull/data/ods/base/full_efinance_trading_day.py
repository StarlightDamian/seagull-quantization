# -*- coding: utf-8 -*-
"""
@Date: 2024/5/5 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: full_efinance_trading_day.py
@Description: 交易日(ods_base_full_efinance_trading_day)
"""
import os
from datetime import datetime 

import pandas as pd
import adata

from __init__ import path
from utils import utils_database, utils_data, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


class TradingDay:
    def __init__(self):
        ...
    
    def ods_incr_efinance_trading_day_1(self, year):
        # 2005年开始至今的交易日
        # 不能翻墙
        logger.info(f'trading_year: {year}')
        trading_day_df = adata.stock.info.trade_calendar(year=int(year))
        return trading_day_df
    
    def ods_incr_efinance_trading_day(self):
        two_years_later = str(datetime.today().year + 2)
        years = pd.date_range(start='2005', end=two_years_later, freq='YE').year
        trading_day_df = pd.concat([self.ods_incr_efinance_trading_day_1(year) for year in years], ignore_index=True)
        utils_data.output_database(trading_day_df, filename='ods_info_incr_efinance_trading_day', if_exists='replace')

        #trading_day_df = years_df.groupby('year').apply(self.ods_incr_efinance_trading_day_1)
        #trading_day_df = adata.stock.info.trade_calendar(year=2025)  
        #print(df)
        return trading_day_df
    
    def dwd_incr_trading_day(self, fliename='ods_info_incr_efinance_trading_day'):
        with utils_database.engine_conn('postgre') as conn:
            trading_day_df = pd.read_sql(fliename, con=conn.engine)
        trading_day_df = trading_day_df.rename(columns={'trade_date': 'date',
                                                        'day_week': 'week'})
        utils_data.output_database(trading_day_df, filename='dwd_info_incr_trading_day', if_exists='replace')

    def dwd_incr_baostoc_trading_day_demo():
        """
        废弃方案
        baostock获取交易日demo,无法获取明年的交易日
        """
        import baostock as bs
        import pandas as pd
        bs.login()
        rs = bs.query_trade_dates(start_date="2017-01-01", end_date="2035-06-30")
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(result)
        bs.logout()
        
    def pipeline(self):
        # 更新周期: 年或月
        self.ods_incr_efinance_trading_day()
        self.dwd_incr_trading_day()
        
        
if __name__ == '__main__':
    trading_day = TradingDay()
    trading_day.pipeline()
    