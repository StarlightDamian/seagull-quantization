# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 10:12
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_sh_sz_fetch5.py
@Description: 
"""
# -*- coding: utf-8 -*-
"""
@Date:   2025/07/07
@Author: Damian
@File:   stock_incr_baostock_sh_sz_fetch_multiproc.py
@Desc:   A股5分钟K线增量拉取，多进程并行、批量后处理、全流程日志和进度监控
"""
# -*- coding: utf-8 -*-
"""
@Date:   2025/07/07
@Author: Damian
@File:   stock_incr_baostock_sh_sz_fetch_multiproc.py
@Desc:   A股5分钟K线增量拉取，多进程并行、批量写库（每100条）、全流程日志和进度监控
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

# 日志配置
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f"{PATH}/log/{log_filename}.log")


# 进程内登录及初始化
def init_worker():
    bs.login()
    utils_log.logger_remove_console()


# 单任务拉取，并将日志写入 ods_base_log
def fetch_task(p: Dict[str, str]) -> pd.DataFrame:
    code, ds, de = p['code'], p['date_start'], p['date_end']
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,time,code,open,high,low,close,volume",
            start_date=ds,
            end_date=de,
            frequency="5",
            adjustflag="2"
        )
        time.sleep(0.5)
        df = rs.get_data()
        volume = len(df)
        error_code = rs.error_code
        # 写日志表
        log_df = pd.DataFrame([{
            'error_code': error_code,
            'full_code':  code,
            'volume':     volume,
            'date_start': ds,
            'date_end':   de,
            'freq_code':  5,
            'adj_code':   1,
            'insert_time': datetime.now().strftime('%F %T')
        }])
        utils_data.output_database_large(log_df, filename='ods_base_log', if_exists='append')
        # 进度日志
        days = volume / 48.0
        if error_code == '0' and volume != 0:
            print(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")
        elif error_code == '0' and volume == 0:
            print(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")
        else:
            print(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")

        if error_code == '0' and volume != 0:
            logger.success(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")
        elif error_code == '0' and volume == 0:
            logger.info(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")
        else:
            logger.warning(f"[{code}] {volume} bars (~{days:.1f} days) fetched, error_code={error_code}")
        return df
    except Exception as e:
        logger.error(f"[ERROR] {code} {ds}->{de} Exception: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    overall_start = time.time()
    logger.info(f"[MAIN BEGIN] {datetime.now()} starting multiprocess fetch")

    # 参数准备
    with utils_database.engine_conn('POSTGRES') as conn:
        codes = pd.read_sql("SELECT code FROM ods_info_stock_incr_baostock", con=conn.engine)['code'].tolist()
    batch = utils_time.make_param_grid(
        date_start='2024-01-01',
        date_end=str(datetime.today().date()),
        window_days=60,
        code=codes
    )
    dfp = pd.DataFrame(batch)
    dfp['date_end'] = (pd.to_datetime(dfp['date_end']) - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    params = dfp.to_dict('records')

    total_tasks = len(params)
    cpu_count = max(1, mp.cpu_count() - 1)
    logger.info(f"Use {cpu_count} processes, total tasks={total_tasks}")

    # 并行拉取并批量写入（每100条）
    with mp.Pool(processes=cpu_count, initializer=init_worker) as pool:
        buffer = []
        for idx, df in enumerate(pool.imap_unordered(fetch_task, params), start=1):
            buffer.append(df)
            # 每100条写库一次
            if (idx % 100 == 0) and (idx != 0):
                df_list = [d for d in buffer if not d.empty]
                if df_list:
                    chunk = pd.concat(df_list, ignore_index=True)
                    if not chunk.empty:
                        chunk = split_baostock_code(chunk)
                        chunk = chunk.astype({
                            'full_code': 'category',
                            'date':      'category',
                            'time':      'category',
                            'open':      'float32',
                            'high':      'float32',
                            'low':       'float32',
                            'close':     'float32',
                            'volume':    'int64'
                        })
                        chunk[['open','high','low','close']] = chunk[['open','high','low','close']].round(4)
                        utils_data.output_database_large(
                            chunk,
                            filename='ods_ohlc_stock_incr_baostock_sh_sz_minute',
                            if_exists='append'
                        )
                logger.info(f"Batch {idx//100} written ({idx}/{total_tasks})")
                buffer.clear()
        # 处理剩余不足100的
        if buffer:
            df_list = [d for d in buffer if not d.empty]
            if df_list:
                remainder = pd.concat([d for d in buffer if not d.empty], ignore_index=True)
                if not remainder.empty:
                    remainder = split_baostock_code(remainder)
                    remainder = remainder.astype({
                        'full_code': 'category',
                        'date':      'category',
                        'time':      'category',
                        'open':      'float32',
                        'high':      'float32',
                        'low':       'float32',
                        'close':     'float32',
                        'volume':    'int64'
                    })
                    remainder[['open','high','low','close']] = remainder[['open','high','low','close']].round(4)
                    utils_data.output_database_large(
                        remainder,
                        filename='ods_ohlc_stock_incr_baostock_sh_sz_minute',
                        if_exists='append'
                    )
                logger.info(f"Final batch written ({total_tasks}/{total_tasks})")

    logger.info(f"[MAIN END] total elapsed {time.time()-overall_start:.2f}s at {datetime.now()}")
    bs.logout()
    logger.info('logout success')
