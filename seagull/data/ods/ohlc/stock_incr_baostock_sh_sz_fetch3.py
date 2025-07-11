# -*- coding: utf-8 -*-
"""
@Date:   2025/07/06
@Author: Damian
@File:   stock_incr_baostock_sh_sz_fetch_optimized.py
@Desc:   A股5分钟K线增量拉取，高效并发、串行Baostock调用、批量后处理
"""

import os
import time
import threading
import asyncio
from datetime import datetime
from typing import List, Dict

import pandas as pd
import baostock as bs

from seagull.utils.api.utils_api_baostock import split_baostock_code
from seagull.utils import utils_data, utils_database, utils_log, utils_time
from seagull.settings import PATH

# 日志配置
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f"{PATH}/log/{log_filename}.log")

class BaostockFetcher:
    """同步调用 Baostock，内部串行化、重试与退避"""
    _lock = threading.Lock()

    def __init__(self, max_attempts: int = 3, backoff: float = 0.5):
        self.max_attempts = max_attempts
        self.backoff = backoff

    def fetch(self, code: str, start: str, end: str) -> pd.DataFrame:
        with BaostockFetcher._lock:
            for i in range(1, self.max_attempts + 1):
                try:
                    rs = bs.query_history_k_data_plus(
                        code,
                        fields="date,time,code,high,low,close,volume",
                        start_date=start,
                        end_date=end,
                        frequency="5",
                        adjustflag="2"
                    )
                    return rs.get_data()
                except Exception as e:
                    logger.warning(f"[Baostock] {code} attempt {i} failed: {e}")
                    time.sleep(self.backoff * i)
            logger.error(f"[Baostock] {code} all {self.max_attempts} attempts failed")
            return pd.DataFrame()

class YourFetcher:
    """异步并发控制，批量拉取与后处理"""
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.sync_fetcher = BaostockFetcher()

    async def _worker(self, p: Dict[str, str], sem: asyncio.Semaphore) -> pd.DataFrame:
        async with sem:
            code, ds, de = p["code"], p["date_start"], p["date_end"]
            df = await asyncio.to_thread(self.sync_fetcher.fetch, code, ds, de)
            if self.delay:
                await asyncio.sleep(self.delay)
            return df

    def process_batch(self, params: List[Dict[str, str]], max_concurrency: int = 20) -> pd.DataFrame:
        t0 = time.time()
        logger.info(f"[BEGIN] total tasks={len(params)} at {datetime.now()}")

        async def _run():
            sem = asyncio.Semaphore(max_concurrency)

            logger.info("  → scheduling tasks...")
            tasks = [self._worker(p, sem) for p in params]

            scheduler_end = time.time()
            logger.info(f"    scheduling done in {scheduler_end-t0:.2f}s, gathering results...")

            results = []
            completed = 0
            for fut in asyncio.as_completed(tasks):
                df = await fut
                results.append(df)
                completed += 1
                pct = completed / len(params) * 100
                logger.info(f"    progress: {completed}/{len(params)} ({pct:.1f}%)")

            gather_end = time.time()
            logger.info(f"    gathering completed in {gather_end-scheduler_end:.2f}s")

            # 合并后处理一次
            dfs = [df for df in results if not df.empty]
            if not dfs:
                return pd.DataFrame()
            merge_start = time.time()
            df_all = pd.concat(dfs, ignore_index=True)
            logger.info(f"  → merged {len(dfs)} frames in {time.time()-merge_start:.2f}s")

            post_start = time.time()
            df_all = split_baostock_code(df_all)
            df_all = df_all.astype({
                "full_code": "category",
                "date":      "category",
                "time":      "category",
                "high":      "float32",
                "low":       "float32",
                "close":     "float32",
               # "volume":    "int64"
            })
            df_all[["high","low","close"]] = df_all[["high","low","close"]].round(4)
            logger.info(f"  → post-processing took {time.time()-post_start:.2f}s")

            db_start = time.time()
            utils_data.output_database_large(
                df_all,
                filename="ods_ohlc_stock_incr_baostock_sh_sz_minute",
                if_exists="append"
            )
            logger.info(f"  → output to database took {time.time()-db_start:.2f}s")
            return df_all

        df_final = asyncio.run(_run())
        logger.info(f"[END] total elapsed {time.time()-t0:.2f}s at {datetime.now()}")
        return df_final


if __name__ == "__main__":
    bs.login()
    with utils_database.engine_conn("POSTGRES") as conn:
        codes = pd.read_sql("SELECT code FROM ods_info_stock_incr_baostock", con=conn.engine)["code"].tolist()
    batch = utils_time.make_param_grid(
        date_start="2024-01-01",
        date_end=str(datetime.today().date()),
        window_days=70,
        code=codes
    )
    dfp = pd.DataFrame(batch)
    dfp["date_end"] = (pd.to_datetime(dfp["date_end"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    dfp = dfp.sample(n=100, random_state=42)
    params = dfp.to_dict("records")

    fetcher = YourFetcher(delay=0.0)
    result = fetcher.process_batch(params, max_concurrency=20)

    bs.logout()

#74.17s/100, max_concurrency=5,window_days=60,
#40.88s-50.04s-90.58/100, max_concurrency=20,window_days=60,
#56.22s/100, max_concurrency=30,window_days=60,
#82.48s-97.80s , max_concurrency=20,window_days=70,