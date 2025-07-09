# -*- coding: utf-8 -*-
"""
@Date: 2025/7/5 23:39
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz_fetch.py
@Description: 最多55天

            date               time       code  ...        low      close    volume
0     2024-07-02  20240702093500000  sz.399998  ...  2406.8900  2421.0000  83480400
1     2024-07-02  20240702094000000  sz.399998  ...  2398.6400  2398.6400  30231500
2     2024-07-02  20240702094500000  sz.399998  ...  2396.8200  2399.0100  28477500
3     2024-07-02  20240702095000000  sz.399998  ...  2387.4800  2391.2500  26631700
4     2024-07-02  20240702095500000  sz.399998  ...  2383.4700  2384.2200  25129400
         ...                ...        ...  ...        ...        ...       ...
2635  2024-09-18  20240918144000000  sz.399998  ...  1916.1000  1918.1500  10846500
2636  2024-09-18  20240918144500000  sz.399998  ...  1916.0800  1917.3800  11645900
2637  2024-09-18  20240918145000000  sz.399998  ...  1914.1700  1915.8200  11786600
2638  2024-09-18  20240918145500000  sz.399998  ...  1914.3900  1916.7000  14456700
2639  2024-09-18  20240918150000000  sz.399998  ...  1915.8300  1918.8700  12088300

[2640 rows x 7 columns]
"""
import asyncio
import argparse
from datetime import datetime  # , timedelta

import pandas as pd
import baostock as bs
from seagull.settings import PATH
from seagull.utils.api.utils_api_baostock import split_baostock_code
from seagull.utils import utils_time, utils_data, utils_thread, utils_database, utils_character, utils_log
#from seagull.data.ods.ohlc.stock_incr_baostock_sh_sz import OdsOhlcStockIncrBaostockShSz
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
import pandas as pd
from typing import List, Dict
import time

import os

import baostock as bs
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class OdsOhlcStockIncrBaostockShSz:
    """
    A股的K线数据，全量历史数据接口
    """
    def __init__(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            self.ods_stock_base_df = pd.read_sql("select code from ods_info_stock_incr_baostock", con=conn.engine)  # 获取指数、股票数据
            # code = pd.read_sql("select distinct code from ods_ohlc_incr_baostock_stock_sh_sz_daily", con=conn.engine)
            # self.ods_stock_base_df = self.ods_stock_base_df[~(self.ods_stock_base_df.code.isin(code))]

        # 创建异步HTTP会话
        self.session = None

    def stock_sh_sz_1(self, code,
                            date_start,
                            date_end,
                            fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
                            frequency='d',
                            adjustflag='3'):
        #code = substring.name
        logger.info(f'start code: {code}{fields}{date_start} - {date_end} | frequency: {frequency} | adjustflag: {adjustflag}')
        k_rs = bs.query_history_k_data_plus(code,
                                            fields=fields,
                                            start_date=date_start,
                                            end_date=date_end,
                                            frequency=frequency,
                                            adjustflag=adjustflag
                                            )
        try:
            logger.info(f'date_start: {date_start}| date_end: {date_end}')
            data_df = k_rs.get_data()
        except:
            logger.error(code)
        if data_df.empty:
            logger.warning(f'{code} empty')
        else:
            logger.info(f'{code} {data_df.shape}')
            return data_df
class YourFetcher(OdsOhlcStockIncrBaostockShSz):
    def __init__(self, delay: float = 0, max_concurrency=20):
        """
        :param delay: 每次请求后睡眠的秒数，默认 0（不睡眠）
        """
        self.delay = delay
        self._sem = asyncio.Semaphore(max_concurrency)
        #self._executor = ThreadPoolExecutor(max_concurrency)

    # 改成接收三参：date_start, date_end, code
    async def async_fetch_minute(
        self,
        code: str,
        date_start: str,
        date_end: str,
    ) -> pd.DataFrame:
        async with self._sem:
            # await asyncio.sleep(0.1), 这里模拟网络 I/O
            df = self.stock_sh_sz_1(
                code=code,
                date_start=date_start,
                date_end=date_end,
                fields='date,time,code,open,high,low,close,volume',
                frequency='5',
                adjustflag='2'
            )
            # 可配置的延迟，防止跑太快被限流
            if self.delay and self.delay > 0:
                await asyncio.sleep(self.delay)

            return df

    # 主线程里调用，统一传一个参数字典列表
    def process_batch(
        self,
        params: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        params: [
            {'date_start': '2025-07-01', 'date_end': '2025-07-10', 'code': '600000.SH'},
            {'date_start': '2025-07-01', 'date_end': '2025-07-10', 'code': '000001.SZ'},
            ...
        ]
        """
        # 构造协程任务
        t0 = time.time()
        async def run_all():
            tasks = [
                self.async_fetch_minute(
                    p['code'],
                    p['date_start'],
                    p['date_end']
                )
                for p in params
            ]
            # 并发执行
            dfs = await asyncio.gather(*tasks, return_exceptions=False)
            # 丢到数据库
            df_all = pd.concat(dfs, ignore_index=True)

            if not df_all.empty:
                df_all = split_baostock_code(df_all)

                # 'date,time,code,open,high,low,close,volume'
                df_all = df_all.astype({"full_code": "category",
                                        "date": "category",
                                        "time": "category",
                                        "open": "float32",
                                        "high": "float32",
                                        "low": "float32",
                                        "close": "float32",
                                        # "volume": "low",
                                       # "freq": "category",
                                       # "adj_type": "category"
                                        })
                df_all[['open', 'high', 'low', 'close']] = df_all[['open', 'high', 'low', 'close']].round(4)
                df_all = df_all[["date", "time", "full_code", "open", "high", "low", "close", "volume",
                                # "adj_type", "freq", "primary_key"
                                 ]]
                utils_data.output_database_large(df_all,
                                                 filename='ods_ohlc_stock_incr_baostock_sh_sz_minute',
                                                 if_exists='append'
                                                 )
                logger.info(f"[END] total elapsed {time.time() - t0:.2f}s at {datetime.now()}")

            return df_all

        # 同步调用
        return asyncio.run(run_all())


# 用法示例
if __name__ == '__main__':
    lg = bs.login()
    fetcher = YourFetcher(delay=0)

    # 先查出你要的 code 列表
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_codes = pd.read_sql(
            "SELECT code FROM ods_info_stock_incr_baostock",
            con=conn.engine
        )['code'].tolist()
    # 根据每个 code 和各自的起止日期构造参数字典
    batch_params = utils_time.make_param_grid(
        # date_start="2019-01-01",
        date_start="2024-04-01",
        date_end="2025-07-08",
        window_days=70,
        code=stock_codes
    )
    print(len(batch_params))  # 481984
    df = pd.DataFrame(batch_params)  # len(df.code.unique())  # 7088
    df['date_end'] = pd.to_datetime(df['date_end'], format='%Y-%m-%d') - pd.Timedelta(days=1)
    df['date_end'] = df['date_end'].dt.strftime('%Y-%m-%d')

    #df = df.sample(n=100, random_state=42)
    batch_params = [x for x in df.T.to_dict().values()]
    fetcher.process_batch(batch_params)

    bs.logout()

#33.24s-54.14s-129.98s/100