# -*- coding: utf-8 -*-
"""
@Date: 2024/4/30 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_thread.py
@Description: 多线程
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def thread(grouped, fun, max_workers=32, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list = list(executor.map(lambda group: fun(group, *args, **kwargs), [group for _, group in grouped]))
    df = pd.concat(data_list)
    return df


def process(grouped, fun, max_workers=18, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        data_list = list(executor.map(lambda group: fun(group, *args, **kwargs), [group for _, group in grouped]))
    df = pd.concat(data_list)
    return df


def pool():
    pbar = tqdm(total=len(stocks))
    
    def do_work(code: str, klt: KLT):
        persist_stock(code, klt)
        pbar.update(1)
    
    # 并行下载行情数据。
    pool = Pool(8)
    for stock in stocks:
        pool.apply_async(func=do_work, args=(stock, klt))
    pool.close()
    pool.join()


if __name__ == '__main__':
    from seagull.settings import PATH
    from seagull.utils import utils_database
    
    def prev_close(df: pd.DataFrame) -> pd.DataFrame:
        # df = df.sort_values(by='date', ascending=True)
        df[['prev_close']] = df[['close']].shift(1)
        return df
    
    with utils_database.engine_conn("POSTGRES") as conn:
        #etf_df = pd.read_sql("select * from dwd_ohlc_incr_stock_daily where board_type='ETF'", con=conn.engine)
        etf_df = pd.read_sql("ods_ohlc_incr_efinance_portfolio_daily", con=conn.engine)
        bj_df = pd.read_sql("ods_ohlc_incr_efinance_stock_bj_daily", con=conn.engine)
    df = pd.concat([bj_df, etf_df], axis=0)
    grouped = df.groupby('full_code')
    result_df = thread(grouped, prev_close, max_workers=4)
    print(result_df)