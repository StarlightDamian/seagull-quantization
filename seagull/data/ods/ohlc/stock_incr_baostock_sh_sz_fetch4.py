# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 0:15
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz_fetch4.py
@Description: 多线程
A股5分钟K线增量拉取，多进程并行、批量后处理、全流程计时


1.因为一个参数跑一次API会有2000条数据，每个子进程获取到100次API返回的数据，无论成功还是失败，都写入数据库ods_ohlc_stock_incr_baostock_sh_sz_minute，append
2.每次写入库时把日志数据写入日志表，ods_base_log，append,字段：error_code、full_code、volume、date_start、date_end、freq_code=5,adj_code=1,
3.“            logger.info(f"    scheduling done in {scheduler_end-t0:.2f}s, gathering results...")
            logger.info(f"  → merged {len(dfs)} frames in {time.time()-merge_start:.2f}s")”参考这个写法，在子进程跑的时候，我希望获取到“当前数据量/数据总量“，和百分比进度。因为这个是五分钟k线，我希望子进程 ”数据量/48“，统计得到获取了多少天的数据。
4.在整个主进程最外层套一个time，来记录跑了多少次API，跑了多长时间，精确到秒。
"""
import os
import time
import multiprocessing as mp
from datetime import datetime
from typing import List, Dict

import pandas as pd
import baostock as bs

from seagull.utils.api.utils_api_baostock import split_baostock_code
from seagull.utils import utils_data, utils_database, utils_log, utils_time
from seagull.settings import PATH
import time

# log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_console()


# 进程内登录
def init_worker():
    bs.login()
    # 子进程移除 console sink（只剩文件输出）
    utils_log.logger_remove_console()


# 单任务拉取函数
def fetch_task(p: Dict[str, str]) -> pd.DataFrame:
    code = p['code']
    ds = p['date_start']
    de = p['date_end']
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,time,code,open,high,low,close,volume",
            start_date=ds,
            end_date=de,
            frequency="5",
            adjustflag="2"
        )
        df = rs.get_data()
        time.sleep(0.1)
        volume = df.shape[0]
        if df.empty and (rs.error_code == '0'):
            logger.info(f"error_code: 0 |volume:0|'{code}',start_date='{ds}', end_date='{de}'")
        elif (not df.empty) and (rs.error_code == '0'):
            logger.success(f"error_code: 0 |volume:{volume}|'{code}',start_date='{ds}', end_date='{de}'")
        elif rs.error_code != '0':
            logger.warning(f"error_code: {rs.error_code}|volume:{volume}|'{code}',start_date='{ds}', end_date='{de}'")
        else:
            logger.warning(f"else |error_code: {rs.error_code}|volume:{volume}|'{code}',start_date='{ds}', end_date='{de}'")
        return df

    except Exception as e:
        logger.error(f" {code} {ds}->{de}: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    start_time = time.time()
    logger.info(f"[BEGIN] {datetime.now()} Fetch start")

    # 获取股票列表和时间窗口
    with utils_database.engine_conn('POSTGRES') as conn:
        codes = pd.read_sql("SELECT code FROM ods_info_stock_incr_baostock", con=conn.engine)['code'].tolist()
    batch = utils_time.make_param_grid(
        date_start='2024-01-01',
        date_end=str(datetime.today().date()),
        window_days=60,
        code=codes
    )
    # 调整 date_end
    dfp = pd.DataFrame(batch)
    dfp['date_end'] = (pd.to_datetime(dfp['date_end']) - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    # dfp = dfp.sample(n=100, random_state=42)
    params = dfp.to_dict('records')

    # 并行拉取
    proc_start = time.time()
    cpu_count = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {cpu_count} processes for fetch, total tasks={len(params)}")

    with mp.Pool(processes=cpu_count, initializer=init_worker) as pool:
        dfs = pool.map(fetch_task, params)
    logger.info(f"Fetch tasks completed in {time.time()-proc_start:.2f}s")

    # 合并与后处理
    merge_start = time.time()
    df_all = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    df_all = split_baostock_code(df_all)
    df_all = df_all.astype({
        'full_code': 'category',
        'date':      'category',
        'time':      'category',
        'open':      'float32',
        'high':      'float32',
        'low':       'float32',
        'close':     'float32',
        'volume':    'int64'
    })
    df_all[['open', 'high', 'low', 'close']] = df_all[['open', 'high', 'low', 'close']].round(4)
    logger.info(f"Post-processing took {time.time()-merge_start:.2f}s")

    # 写库
    db_start = time.time()
    utils_data.output_database_large(
        df_all,
        filename='ods_ohlc_stock_incr_baostock_sh_sz_minute',
        if_exists='append'
    )
    logger.info(f"Database output took {time.time()-db_start:.2f}s")

    logger.info(f"[END] total elapsed {time.time()-start_time:.2f}s at {datetime.now()}")
    bs.logout()
    logger.info('logout success')

# 1000条
# 2025-07-08 21:32:00.031 | INFO     | __main__:<module>:89 - Fetch tasks completed in 240.41s
# 2025-07-08 21:32:03.063 | INFO     | __main__:<module>:105 - Post-processing took 3.03s

#100
#2025-07-08 21:37:17.409 | INFO     | __main__:<module>:89 - Fetch tasks completed in 24.00s
#2025-07-08 21:37:17.671 | INFO     | __main__:<module>:105 - Post-processing took 0.26s