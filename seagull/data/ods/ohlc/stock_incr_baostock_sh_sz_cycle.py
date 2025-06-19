# -*- coding: utf-8 -*-
"""
@Date: 2023/8/8 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz_cycle.py
@Description: 获取指定日期全部股票的日K线数据(ods/ohlc/stock_incr_baostock_sh_sz_cycle)
code_name 不属于特征，在这一层加入
frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。

断网导致部分数据没跑出来，删除多余数据
select date,count(1) from ods_ohlc_incr_baostock_stock_sh_sz_minute group by date order by date desc
delete from ods_ohlc_incr_baostock_stock_sh_sz_minute where date > '2022-01-05'
"""
import asyncio
import argparse
from datetime import datetime  # , timedelta

import pandas as pd

from seagull.utils import utils_time, utils_data, utils_thread
from data.ods.ohlc import ods_ohlc_incr_baostock_stock_sh_sz_api


class OdsIncrBaostockStockShSzCycle(ods_ohlc_incr_baostock_stock_sh_sz_api.OdsIncrBaostockStockShSzApi):
    """
    ODS层_baostock接口_证券_上海_深圳_每日数据_接口：A股的K线数据，全量历史数据接口
    """
    def __init__(self):
        super().__init__()
    
    def stock_sh_sz_daily(self, date_start="1990-01-01", date_end="2100-01-01"):
        date_start = utils_data.maximum_date_next(table_name='ods_ohlc_incr_baostock_stock_sh_sz_daily', field_name='date')
        stock_sh_sz_daily_df = self.stock_sh_sz(date_start=date_start, date_end=date_end)
        utils_data.output_database(stock_sh_sz_daily_df,
                                   filename='ods_ohlc_incr_baostock_stock_sh_sz_daily')
        # return stock_sh_sz_daily_df

    def stock_sh_sz_minute_date_1(self, subtable, fields='date,time,code,high,low,close,volume'):
        #fields='date,time,code,open,high,low,close,volume,amount'
        #print(subtable.name.)
        date = str(subtable.name)
        date_start = utils_time.date_plus_days(date, days=1)
        date_end = utils_time.date_plus_days(date, days=250)  # 和freq='100D'匹配
        #if utils_database.table_exists('ods_ohlc_incr_baostock_stock_sh_sz_minute'):
        #    sql=f'SELECT max(date) FROM ods_ohlc_incr_baostock_stock_sh_sz_minute_api BETWEEN {date_start} AND {date_end}'
        #    date_start = utils_data.maximum_date(table_name='ods_ohlc_incr_baostock_stock_sh_sz_minute', sql=sql)
        
        stock_sh_sz_minute_df = self.stock_sh_sz(date_start=date_start,
                                                 date_end=date_end,
                                                 fields=fields,
                                                 frequency='5')
        if not stock_sh_sz_minute_df.empty:
            utils_data.output_database_large(stock_sh_sz_minute_df,
                                       filename='ods_ohlc_incr_baostock_stock_sh_sz_minute',
                                       if_exists='append',
                                       )
        
    def stock_sh_sz_minute(self, date_start, date_end):
        daily_dates = pd.date_range(start=date_start, end=date_end, freq='250d')  # bs.query_history_k_data_plus返回数据量不超过10000,分日获取
        daily_dates_df = pd.DataFrame(daily_dates, columns=['date'])
        daily_dates_df.date = daily_dates_df.date.astype(str)
        #daily_dates_df.groupby('date').apply(self.stock_sh_sz_minute_date_1, fields='date,time,code,high,low,close,volume')
        
    def stock_sh_sz_weekly_monthly(self, date_start, date_end):
        # 周期型数据,和日线的参数有些不同
        date_start = utils_data.maximum_date(table_name='ods_ohlc_incr_baostock_stock_sh_sz_cycle', field_name='date')
        stock_sh_sz_cycle_df = self.stock_sh_sz(date_start=date_start, date_end=date_end, frequency='w')  # weekly
        stock_sh_sz_cycle_df = self.stock_sh_sz(date_start=date_start, date_end=date_end, frequency='m')  # monthly
        utils_data.output_database(stock_sh_sz_cycle_df,
                                   filename='ods_ohlc_incr_baostock_stock_sh_sz_cycle')
    # 异步执行单日抓取的示例
    async def async_fetch_date(self, date: str) -> pd.DataFrame:
        #await asyncio.sleep(0.1)  # 模拟网络 I/O
        #date = str(subtable.name)
        date_start = utils_time.date_plus_days(date, days=1)
        date_end = utils_time.date_plus_days(date, days=250)  # 和freq='100D'匹配
        # if utils_database.table_exists('ods_ohlc_incr_baostock_stock_sh_sz_minute'):
        #    sql=f'SELECT max(date) FROM ods_ohlc_incr_baostock_stock_sh_sz_minute_api BETWEEN {date_start} AND {date_end}'
        #    date_start = utils_data.maximum_date(table_name='ods_ohlc_incr_baostock_stock_sh_sz_minute', sql=sql)

        stock_sh_sz_minute_df = self.stock_sh_sz(date_start=date_start,
                                                 date_end=date_end,
                                                 fields=fields,
                                                 frequency='5')

        return stock_sh_sz_minute_df

    # 在主线程中运行协程批量抓取
    async def fetch_batch(self, dates):
        tasks = [self.async_fetch_date(d) for d in dates]
        dfs = await asyncio.gather(*tasks)
        return pd.concat(dfs, ignore_index=True)

    # 子线程调用的同步接口——在这里顺手写库
    def process_batch(self, dates):
        batch_df = asyncio.run(self.fetch_batch(dates))
        if not batch_df.empty:
            utils_data.output_database_large(
                batch_df,
                filename='ods_ohlc_incr_baostock_stock_sh_sz_minute',
                if_exists='append'
            )
        return batch_df
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for backtesting')
    #parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_start', type=str, default='2021-09-27', help='Start time for backtesting')
    #parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-05-23', help='End time for backtesting')
    #parser.add_argument('--update_type', type=str, default='incr', help='Data update method')
    parser.add_argument('--freq_type', type=str, default='minute', help='daily or minute')
    args = parser.parse_args()
    
    ods_incr_baostock_stock_sh_sz_cycle = OdsIncrBaostockStockShSzCycle()
    date_end = args.date_end if args.date_end !='' else datetime.now().strftime("%F")
    if args.freq_type=='daily':
        ods_incr_baostock_stock_sh_sz_cycle.stock_sh_sz_daily(date_start=args.date_start,
                                                              date_end=args.date_end)  # 每日沪深
    elif args.freq_type=='minute':
        daily_dates = pd.date_range(start=args.date_start, end=args.date_end, freq='250d')  # bs.query_history_k_data_plus返回数据量不超过10000,分日获取
        daily_dates_df = pd.DataFrame(daily_dates, columns=['date'])
        daily_dates_df.date = daily_dates_df.date.astype(str).tolist()
        df_all = utils_thread.thread(daily_dates_df, ods_incr_baostock_stock_sh_sz_cycle.process_batch, max_workers=32)

        # ods_incr_baostock_stock_sh_sz_cycle.stock_sh_sz_minute(date_start=args.date_start,
        #                                                        date_end=args.date_end)
