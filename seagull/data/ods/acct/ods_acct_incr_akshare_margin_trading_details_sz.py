# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 23:02:29 2025

@author: Damian
深圳两融数据明细(ods_acct_incr_akshare_margin_trading_details_sz)
['证券代码', '证券简称', '融资买入额', '融资余额', '融券卖出量', '融券余量',
 '融券余额', '融资融券余额'], 
"""
import os
import argparse

import pandas as pd
import akshare as ak

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

def _apply_margin_trading_details_sz(sub):
    try:
        date = sub.name.replace('-','')
        df = ak.stock_margin_detail_szse(date=date)
        logger.success(date)
        return df
    except:
        logger.warning(date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-04-01', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='full', help='Data update method')
    parser.add_argument('--filename', type=str, default='ods_acct_incr_akshare_margin_trading_sh_details', help='Database table name')
    args = parser.parse_args()
    
    with utils_database.engine_conn("POSTGRES") as conn:
        trading_day_df = pd.read_sql(f"select * from dwd_info_incr_trading_day where date>='{args.date_start}' and date<'{args.date_end}' and trade_status=1", con=conn.engine)
        
    margin_trading_details_sz_df = trading_day_df.groupby('date').apply(_apply_margin_trading_details_sz)
    
    utils_data.output_database_large(margin_trading_details_sz_df,
                                     filename='ods_acct_incr_akshare_margin_trading_details_sz',
                                     )
    