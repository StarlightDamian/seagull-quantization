# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:13:53 2024

@author: awei
(dwd_feat_incr_macd)
需要区分adj_type和freq
"""
import os
import argparse
from datetime import datetime

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data
from feature import macd
from finance import finance_trading_day

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def pipeline(date_start, date_end):
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_daily_df = pd.read_sql(f"""
            select 
                full_code,
                date,
                close,
                volume,
                turnover,
                turnover_pct,
                primary_key
            from
                dwd_ohlc_incr_stock_daily
            where
                date between '{date_start}' and '{date_end}'
                """, con=conn.engine)
                
    stock_daily_df = stock_daily_df.drop_duplicates('primary_key', keep='first')
    macd_df = macd.pipeline(stock_daily_df,
                              columns=['close', 'volume', 'turnover', 'turnover_pct'],
                              numeric_columns=['close', 'volume', 'turnover']
                              )
    return macd_df

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='full', help='Data update method')
    parser.add_argument('--filename', type=str, default='dwd_feat_incr_macd', help='Database table name')
    args = parser.parse_args()
    
    date_end = args.date_end if args.date_end!='' else datetime.now().strftime("%F")
    if args.update_type=='full':
        df = pipeline(date_start=args.date_start,
                      date_end=date_end)
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='replace')
    elif args.update_type=='incr':
        date_start = utils_data.maximum_date_next(table_name=args.filename)
        trading_day_alignment = finance_trading_day.TradingDayAlignment()
        date_start_prev = trading_day_alignment.shift_day(date_start=date_start, date_num=30)
        
        raw_df = pipeline(date_start=date_start_prev, date_end=date_end)
        df = raw_df[raw_df.date>=date_start]
        
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='append')
        
    
# =============================================================================
#     stock_daily_df.columns=['primary_key', 'date', 'time', 'board_type', 'full_code', 'asset_code',
#            'market_code', 'code_name', 'price_limit_pct', 'open', 'high', 'low',
#            'close', 'prev_close', 'volume', 'turnover', 'adjustflag', 'turnover_pct',
#            'tradestatus', 'pct_chg', 'pe_ttm', 'ps_ttm', 'pcf_ncf_ttm', 'pb_mrq',
#            'is_st', 'insert_timestamp', 'amplitude', 'price_chg', 'freq',
#            'adj_type', 'board_primary_key']
# =============================================================================
# =============================================================================
# df.columns MultiIndex([(12, 26, 9, False, True,        'close'),
#             (12, 26, 9, False, True,       'volume'),
#             (12, 26, 9, False, True,     'turnover'),
#             (12, 26, 9, False, True, 'turnover_pct')],
#            names=['macd_fast_window', 'macd_slow_window', 'macd_signal_window', 'macd_macd_ewm', 'macd_signal_ewm', 'parms'])
# df.index MultiIndex([('000001.sh', '1990-12-19'),
#             ('000001.sh', '1990-12-20'),
#             ('000001.sh', '1990-12-21'),
#             ('000001.sh', '1990-12-24'),
#             ('000001.sh', '1990-12-25'),
#             ('000001.sh', '1990-12-26'),
#             ('000001.sh', '1990-12-27'),
#             ('000001.sh', '1990-12-28'),
#             ('000001.sh', '1990-12-31'),
#             ('000001.sh', '1991-01-02'),
#             ...
#             ('920118.bj', '2024-12-23'),
#             ('920118.bj', '2024-12-24'),
#             ('920118.bj', '2024-12-25'),
#             ('920118.bj', '2024-12-26'),
#             ('920118.bj', '2024-12-27'),
#             ('920118.bj', '2024-12-30'),
#             ('920118.bj', '2024-12-31'),
#             ('920118.bj', '2025-01-02'),
#             ('920118.bj', '2025-01-03'),
#             ('920118.bj', '2024-12-18')],
#            names=['full_code', 'date'], length=56468688)
# =============================================================================
