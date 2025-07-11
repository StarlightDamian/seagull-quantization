# -*- coding: utf-8 -*-
"""
@Date: 2025/7/5 23:39
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz_fetch.py
@Description: A股5分钟K线增量拉取，支持并发控制、串行化Baostock调用、异常重试
完整版
"""
import os
import time
import threading
import asyncio
from typing import List, Dict
import pandas as pd
import baostock as bs
from seagull.utils.api.utils_api_baostock import split_baostock_code
from seagull.utils import utils_data, utils_database, utils_log, utils_time
from seagull.settings import PATH

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class OdsOhlcStockIncrBaostockShSz:
    """
    A股的K线数据拉取（同步方法），包含全局锁、重试、退避
    """
    _baostock_lock = threading.Lock()

    def __init__(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            self.ods_stock_base_df = pd.read_sql(
                "select code from ods_info_stock_incr_baostock",
                con=conn.engine
            )

    def _sync_query(self,
                    code: str,
                    date_start: str,
                    date_end: str,
                    fields: str = 'date,time,code,open,high,low,close,volume',
                    frequency: str = '5',
                    adjustflag: str = '2',
                    max_attempts: int = 5,
                    backoff: float = 0.5
                    ) -> pd.DataFrame:
        """
        串行化Baostock调用 + 重试 + 退避
        """
        with OdsOhlcStockIncrBaostockShSz._baostock_lock:
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"[SYNC] ({attempt}/{max_attempts}) code={code}, {date_start}~{date_end}")
                    rs = bs.query_history_k_data_plus(
                        code,
                        fields=fields,
                        start_date=date_start,
                        end_date=date_end,
                        frequency=frequency,
                        adjustflag=adjustflag
                    )
                    df = rs.get_data()
                    if df.empty:
                        logger.warning(f"[SYNC] {code} empty on attempt {attempt}")
                    else:
                        logger.info(f"[SYNC] {code} got {df.shape} rows on attempt {attempt}")
                    return df
                except (OSError, IOError, zlib.error) as e:
                    logger.warning(f"[SYNC] decompress error for {code} attempt {attempt}: {e}")
                except Exception as e:
                    logger.exception(f"[SYNC] unexpected error for {code} attempt {attempt}: {e}")
                # 退避等待
                time.sleep(backoff * attempt)
            logger.error(f"[SYNC] failed all {max_attempts} attempts for {code}")
            return pd.DataFrame()

    async def async_fetch_minute(self,
                                  code: str,
                                  date_start: str,
                                  date_end: str,
                                  delay: float = 0.1
                                  ) -> pd.DataFrame:
        """
        异步接口：将同步拉取封装到线程池
        """
        df = await asyncio.to_thread(
            self._sync_query,
            code,
            date_start,
            date_end
        )
        if delay > 0:
            await asyncio.sleep(delay)
        return df

class YourFetcher(OdsOhlcStockIncrBaostockShSz):
    """
    并发任务管理，批量拉取接口
    """
    def __init__(self, delay: float = 0.1, max_concurrency: int = 5):
        self.delay = delay
        self.max_concurrency = max_concurrency

    def process_batch(self, params: List[Dict[str, str]]) -> pd.DataFrame:
        """
        同步调用批量拉取：内部并发控制
        params: list of dict with keys code, date_start, date_end
        """
        async def _run_all():
            sem = asyncio.Semaphore(self.max_concurrency)
            async def bounded_fetch(p):
                async with sem:
                    return await self.async_fetch_minute(
                        p['code'], p['date_start'], p['date_end'], self.delay
                    )

            tasks = [bounded_fetch(p) for p in params]
            results = await asyncio.gather(*tasks)
            df_all = pd.concat(results, ignore_index=True) if any(not df.empty for df in results) else pd.DataFrame()
            if not df_all.empty:
                df_all = split_baostock_code(df_all)
                df_all = df_all.astype({
                    'full_code': 'category',
                    'date': 'category',
                    'time': 'category',
                    'high': 'float32',
                    'low': 'float32',
                    'close': 'float32',
                    'volume': 'int64'
                })
                df_all[['high','low','close']] = df_all[['high','low','close']].round(4)
                utils_data.output_database_large(
                    df_all,
                    filename='ods_ohlc_stock_incr_baostock_sh_sz_minute',
                    if_exists='append'
                )
            return df_all

        return asyncio.run(_run_all())

if __name__ == '__main__':
    # 登录一次
    lg = bs.login()
    logger.info('login success')

    # 构造参数
    with utils_database.engine_conn('POSTGRES') as conn:
        codes = pd.read_sql('SELECT code FROM ods_info_stock_incr_baostock', con=conn.engine)['code'].tolist()
    batch_params = utils_time.make_param_grid(
        date_start='2024-02-01', date_end='2025-07-08', window_days=55, code=codes
    )
    sample = pd.DataFrame(batch_params).sample(n=100, random_state=42)
    params = sample.to_dict('records')

    fetcher = YourFetcher(delay=0.1, max_concurrency=5)
    df_out = fetcher.process_batch(params)
    logger.info(f'Fetched total rows: {len(df_out)}')

    bs.logout()
    logger.info('logout success')
