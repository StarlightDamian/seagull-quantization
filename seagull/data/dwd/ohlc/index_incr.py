# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:13:53 2024

@author: awei
(dwd_feat_incr_global_index)

"""
import os
import argparse
from datetime import datetime

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data
from finance import finance_trading_day
from feature import macd

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

                
def pipeline(date_start, date_end):
    full_code_tuple = ('399101.sz', '399102.sz', '000300.sh', '000001.sh', '399106.sz')
    with utils_database.engine_conn("POSTGRES") as conn:
        index_daily_df = pd.read_sql(f"""
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
                full_code in {full_code_tuple}
            and 
                date between '{date_start}' and '{date_end}'
            """, con=conn.engine)
            
    index_daily_df = index_daily_df.drop_duplicates('primary_key', keep='first')
    index_daily_df = macd.pipeline(index_daily_df,
                                   columns=['close', 'volume', 'turnover', 'turnover_pct'],
                                   numeric_columns=['close', 'volume', 'turnover'],
                                   adj_type='pre',
                                   )
    index_daily_df[['prev_close']] = index_daily_df[['close']].shift(1)
    index_daily_df[['close_rate']] = index_daily_df[['close']].div(index_daily_df['prev_close'], axis=0)
    return index_daily_df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='full', help='Data update method')
    parser.add_argument('--filename', type=str, default='dwd_feat_incr_global_index', help='Database table name')
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
        date_start_prev = trading_day_alignment.shift_day(date_start=date_start, date_num=25)
        
        raw_df = pipeline(date_start=date_start_prev, date_end=date_end)
        df = raw_df[raw_df.date >= date_start]
        
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='append')
        
# =============================================================================
# index_daily_df.isna().sum()
# Out[12]: 
# full_code                                0
# date                                     0
# close                                    0
# volume                                   0
# turnover                                 0
# turnover_pct                             0
# primary_key                              0
# close_slope_12_26_9                  29592
# volume_slope_12_26_9                 29592
# turnover_slope_12_26_9               29592
# turnover_pct_slope_12_26_9           29592
# close_acceleration_12_26_9           29592
# volume_acceleration_12_26_9          29592
# turnover_acceleration_12_26_9        29592
# turnover_pct_acceleration_12_26_9    29592
# close_hist_12_26_9                   29592
# volume_hist_12_26_9                  29592
# turnover_hist_12_26_9                29592
# turnover_pct_hist_12_26_9            29592
# close_diff_1                             1
# close_diff_5                             5
# close_diff_30                           22
# volume_diff_1                            1
# volume_diff_5                            5
# volume_diff_30                          22
# turnover_diff_1                          1
# turnover_diff_5                          5
# turnover_diff_30                        22
# turnover_hist_diff_1                 29592
# volume_hist_diff_1                   29592
# close_hist_diff_1                    29592
# prev_close                               1
# close_rate                               1
# insert_timestamp                         0
# dtype: int64
# =============================================================================
