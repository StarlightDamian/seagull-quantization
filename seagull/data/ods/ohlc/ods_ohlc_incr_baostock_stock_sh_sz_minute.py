# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:00:15 2023

@author: awei
获取指定日期全部股票的日K线数据(data_ods_ohlc_incr_baostock_stock_sh_sz_cycle)
code_name 不属于特征，在这一层加入
frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。

断网导致部分数据没跑出来，删除多余数据
select date,count(1) from ods_ohlc_incr_baostock_stock_sh_sz_minute group by date order by date desc
delete from ods_ohlc_incr_baostock_stock_sh_sz_minute where date > '2022-01-05'

if utils_database.table_exists('ods_ohlc_incr_baostock_stock_sh_sz_minute'):
   sql=f'SELECT max(date) FROM ods_ohlc_incr_baostock_stock_sh_sz_minute_api BETWEEN {date_start} AND {date_end}'
   date_start = utils_data.maximum_date(table_name='ods_ohlc_incr_baostock_stock_sh_sz_minute', sql=sql)

"""
import random
import os
import asyncio
import argparse
import time
from datetime import datetime  # , timedelta

import numpy as np
import baostock as bs

from __init__ import path
from utils import utils_data, utils_database, utils_log
from data.ods.ohlc import ods_ohlc_incr_baostock_stock_sh_sz_api

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result

def thread(grouped, fun, max_workers=32, *args, **kwargs):
    """
    Apply `fun(group, *args, **kwargs)` to each DataFrame in `grouped` in parallel,
    then concatenate all results into one DataFrame.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # directly map over the list of DataFrames
        data_list = list(executor.map(lambda group: fun(group, *args, **kwargs), grouped))
    # Concatenate all the returned DataFrames
    df = pd.concat(data_list, ignore_index=True)
    return df


class OdsIncrBaostockStockShSzCycle(ods_ohlc_incr_baostock_stock_sh_sz_api.OdsIncrBaostockStockShSzApi):
    """
    ODS层_baostock接口_证券_上海_深圳_每日数据_接口：A股的K线数据，全量历史数据接口
    """

    def __init__(self):
        super().__init__()

    # 异步执行单日抓取的示例
    async def async_fetch_date(self, params: dict) -> pd.DataFrame:
        # await asyncio.sleep(0.1)  # 模拟网络 I/O
        # date = str(subtable.name)
        # date_start = utils_time.date_plus_days(date, days=1)
        # date_end = utils_time.date_plus_days(date, days=250)  # 和freq='100D'匹配
        code = params['code']
        date_start = params['date_start']
        date_end = params['date_end']
        time.sleep(random.uniform(0.1, 0.5))
        print(f'code: {code} |date_start: {date_start} |date_end: {date_end} |state: start')
        k_rs = bs.query_history_k_data_plus(code,
                                            fields='date,time,code,high,low,close,volume',
                                            start_date=date_start,
                                            end_date=date_end,
                                            frequency='5',
                                            adjustflag='3',
                                            )
        try:
            # logger.info(f'date_start: {date_start}| date_end: {date_end}')
            #time.sleep(1)
            #data_df = k_rs.get_data()
            data_df = get_row_data(k_rs)
        except:
            data_df = pd.DataFrame()
            #logger.error(code)
            print(f'code: {code} |date_start: {date_start} |date_end: {date_end} |state: error')
        if data_df.empty:
            print(f'code: {code} |date_start: {date_start} |date_end: {date_end} |state: empty')
            return data_df
        else:
            print(f'code: {code} |date_start: {date_start} |date_end: {date_end} |state: {data_df.shape}')
            return data_df

        # if utils_database.table_exists('ods_ohlc_incr_baostock_stock_sh_sz_minute'):
        #    sql=f'SELECT max(date) FROM ods_ohlc_incr_baostock_stock_sh_sz_minute_api BETWEEN {date_start} AND {date_end}'
        #    date_start = utils_data.maximum_date(table_name='ods_ohlc_incr_baostock_stock_sh_sz_minute', sql=sql)

       # return stock_sh_sz_minute_df

    # 在主线程中运行协程批量抓取
    async def fetch_batch(self, params_list: list):
        tasks = [self.async_fetch_date(p) for p in params_list]
        dfs = await asyncio.gather(*tasks)
        return pd.concat(dfs, ignore_index=True)

    # 子线程调用的同步接口——在这里顺手写库
    def process_batch(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        params_list = df_sub.to_dict('records')
        batch_df = asyncio.run(self.fetch_batch(params_list))
        if not batch_df.empty:
            utils_data.output_database_large(
                batch_df,
                filename='ods_ohlc_incr_baostock_stock_sh_sz_minute',
                if_exists='append'
            )
        return batch_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for backtesting')
    # parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-05-23', help='End time for backtesting')
    # parser.add_argument('--update_type', type=str, default='incr', help='Data update method')
    parser.add_argument('--cpu_count', type=int, default='4', help='cpu count')
    parser.add_argument('--freq_type', type=str, default='minute', help='daily or minute')
    args = parser.parse_args()

    ods_incr_baostock_stock_sh_sz_cycle = OdsIncrBaostockStockShSzCycle()
    date_end = args.date_end if args.date_end != '' else datetime.now().strftime("%F")
    if args.freq_type == 'daily':
        ods_incr_baostock_stock_sh_sz_cycle.stock_sh_sz_daily(date_start=args.date_start,
                                                              date_end=args.date_end)  # 每日沪深
    elif args.freq_type == 'minute':
        daily_dates = pd.date_range(start=args.date_start, end=args.date_end,
                                    freq='250d')  # bs.query_history_k_data_plus返回数据量不超过10000,分日获取

        with utils_database.engine_conn('postgre') as conn:
            ods_stock_base_df = pd.read_sql("select code from ods_info_incr_baostock_stock_base",
                                            con=conn.engine)  # 获取指数、股票数据

        ods_stock_base_df['date_start'] = [daily_dates] * len(ods_stock_base_df)
        ods_stock_date_df = ods_stock_base_df.explode('date_start')
        ods_stock_date_df['date_start'] = pd.to_datetime(ods_stock_date_df['date_start'])
        ods_stock_date_df['date_end'] = ods_stock_date_df['date_start'] + pd.Timedelta(days=250)

        ods_stock_date_df[['date_start', 'date_end']] = ods_stock_date_df[['date_start', 'date_end']].astype(str)
        stock_date_batches_list = np.array_split(ods_stock_date_df, args.cpu_count)

        bs.login()
        df_all = thread(stock_date_batches_list, ods_incr_baostock_stock_sh_sz_cycle.process_batch, max_workers=args.cpu_count)
        bs.logout()