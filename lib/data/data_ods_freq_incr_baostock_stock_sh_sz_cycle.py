# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:00:15 2023

@author: awei
获取指定日期全部股票的日K线数据(data_ods_freq_incr_baostock_stock_sh_sz_cycle)
code_name 不属于特征，在这一层加入
frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。
"""
import argparse
from datetime import datetime, timedelta

import pandas as pd

from __init__ import path
from base import base_utils
from data import data_utils, data_ods_freq_incr_baostock_stock_api


class odsIncrBaostockStockShSzCycle(data_ods_freq_incr_baostock_stock_api.odsIncrBaostockStockShSzApi):
    """
    ODS层_baostock接口_证券_上海_深圳_每日数据_接口：A股的K线数据，全量历史数据接口
    """
    def __init__(self):
        super().__init__()
    
    def stock_sh_sz_daily(self, date_start="1990-01-01", date_end="2100-01-01"):
        date_start = data_utils.maximum_date(table_name='ods_freq_incr_baostock_stock_sh_sz_daily', field_name='date')
        stock_sh_sz_daily_df = self.stock_sh_sz(date_start=date_start, date_end=date_end)
        data_utils.output_database(stock_sh_sz_daily_df)
        return stock_sh_sz_daily_df
    
    def stock_sh_sz_minute_date_1(self, subtable):
        date = subtable.name
        date_start = base_utils.date_plus_days(date, days=1)
        date_end = base_utils.date_plus_days(date, days=100)  # 和freq='100D'匹配
        sql=f'SELECT max(date) FROM ods_freq_incr_baostock_stock_sh_sz_minute_api BETWEEN {date_start} AND {date_end}'
        date_start = data_utils.maximum_date(table_name='ods_freq_incr_baostock_stock_sh_sz_minute', sql=sql)
        stock_sh_sz_minute_df = self.stock_sh_sz(date_start=date_start,
                                                 date_end=date_end,
                                                 feature='date,time,code,open,high,low,close,volume,amount',
                                                 frequency='5')
        data_utils.output_database(stock_sh_sz_minute_df, filename='ods_freq_incr_baostock_stock_sh_sz_minute')
        
    def stock_sh_sz_minute(self, date_start, date_end):
        daily_dates = pd.date_range(start=date_start, end=date_end, freq='100D')  # bs.query_history_k_data_plus返回数据量不超过10000,分日获取
        daily_dates_df = pd.DataFrame(daily_dates, columns=['date'])
        daily_dates_df.groupby('date').apply(self.stock_sh_sz_minute_date_1)
        
    def stock_sh_sz_weekly_monthly(self, date_start, date_end):
        # 周期型数据,和日线的参数有些不同
        date_start = data_utils.maximum_date(table_name='ods_freq_incr_baostock_stock_sh_sz_cycle', field_name='date')
        stock_sh_sz_minute_df = self.stock_sh_sz(date_start=date_start, date_end=date_end, frequency='w')  # weekly
        stock_sh_sz_minute_df = self.stock_sh_sz(date_start=date_start, date_end=date_end, frequency='m')  # monthly
        data_utils.output_database(stock_sh_sz_minute_df)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-06-01', help='End time for backtesting')
    args = parser.parse_args()
    
    ods_incr_baostock_stock_sh_sz_cycle = odsIncrBaostockStockShSzCycle()
    stock_sh_sz_daily_df = ods_incr_baostock_stock_sh_sz_cycle.stock_sh_sz_daily(date_start=args.date_start, date_end=args.date_end)
    
    #sh_sz_stock_daily_df.to_feather(f'{path}/data/ods_freq_incr_baostock_stock_sh_sz_daily_api.feather') #原生数据
    #k_data_raw_df = pd.read_feather(f'{path}/data/ods_freq_incr_baostock_stock_sh_sz_daily_api.feather')

    
