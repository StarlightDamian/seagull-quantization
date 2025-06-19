# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:50:19 2024

@author: awei
demo_monster


"""
import os
import argparse

import pandas as pd
import numpy as np

from __init__ import path
from utils import utils_database, utils_log, utils_math, utils_data
from finance import finance_limit

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2022-09-20', help='When to start feature engineering')
    parser.add_argument('--date_end', type=str, default='2024-12-27', help='End time for feature engineering')
    args = parser.parse_args()
    
    logger.info(f"""task: feature_engineering
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")
    
    # 获取日期段数据
    with utils_database.engine_conn('postgre') as conn:
        #df = pd.read_sql(f"SELECT * FROM dwd_freq_incr_stock_daily_1 WHERE date BETWEEN '{args.date_start}' AND '{args.date_end}'", con=conn.engine)
        chunks = pd.read_sql("dwd_freq_incr_stock_daily_2", con=conn.engine, chunksize=1_000_000)
        #chunks = pd.read_sql("select * from dwd_freq_incr_stock_daily where board_type in ('北交所','ETF')", con=conn.engine)
    #sz_df = df.loc[df.full_code=='002230.sz']
    
    
    # 逐块处理数据
    for chunk in chunks:
        # 对每个chunk执行操作
        #process(chunk)
        #df_ = chunk.groupby('full_code').apply(finance_limit.limit_prices)
        #chunk = chunk[chunk.board_type.isin(['北交所','ETF'])]
        #if not chunk.empty:
        #chunks = chunks.drop_duplicates('primary_key', keep='first')
        #df_ = chunk.groupby('full_code').apply(finance_limit.limit_prices)
        #df_ = chunks.groupby('full_code').apply(prev_close)
        utils_data.output_database(chunk,
                                   filename='dwd_freq_incr_stock_daily',
                                   if_exists='append')
    # df.loc[df.full_code=='002230.sz',['date','high','close','limit_up']] # 1704 / 427892
    # df_[df_.high>=df_.limit_up]4536/427892
    # df_.loc[df_.is_limit_up==True,['low','high','limit_up','limit_down']] 

