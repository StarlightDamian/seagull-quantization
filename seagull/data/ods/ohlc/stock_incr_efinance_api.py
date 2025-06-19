# -*- coding: utf-8 -*-
"""
@Date: 2024/8/10 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_efinance_api.py
@Description: efinance日频接口(ods/ohlc/stock_incr_efinance_api)
@Update cycle: day
一定要借助清洗后的dwd层的股票基本信息才能关联到到北京对应的股票代码
"""
import os
import argparse
from datetime import datetime

import pandas as pd
import efinance as ef # efinance不能连国际VPN 

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class odsEfinanceStockBjApi():
    def __init__(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            #self.dwd_stock_bj_base_df = pd.read_sql("select * from dwd_info_incr_stock_base where market_code='bj'", con=conn.engine)
            self.dwd_stock_bj_base_df = pd.read_sql("select * from dwd_info_incr_adata_stock_base where market_code='bj'", con=conn.engine)
        logger.info(f'dwd_stock_bj_base_df.shape: {self.dwd_stock_bj_base_df.shape}')
    
    def stock_dict(self, stock_codes, date_start='1900-01-01', date_end='2050-01-01', frequency=101):
        # 优先个股，其次ETF
        logger.info(f'date_start: {date_start}| date_end: {date_end}| len_stock_codes: {len(stock_codes)}')
        beg = date_start.replace('-','')
        end = date_end.replace('-','')
        stock_dict = ef.stock.get_quote_history(stock_codes,
                                                beg=beg,
                                                end=end,
                                                klt=frequency)
        return stock_dict
    
    def stock_minute(self, date_start, date_end, frequency=5):
        # frequency in [5, 15, 30, 60, None]
        stock_minute_dict = self.stock_dict(self.dwd_stock_bj_base_df.asset_code,
                                               date_start=date_start,
                                               date_end=date_end,
                                               frequency=frequency)
        stock_minute_df = pd.concat({k: v for k, v in stock_minute_dict.items()}).reset_index(drop=True)
        #stock_df['frequency'] = 5  # 5分钟
        utils_data.output_database(stock_minute_df,
                                   'ods_ohlc_incr_efinance_stock_bj_daily')
    
    def stock_daily(self, date_start, date_end):
        stock_daily_dict = self.stock_dict(self.dwd_stock_bj_base_df.asset_code,
                                           date_start=date_start,
                                           date_end=date_end
                                           )
        stock_daily_df = pd.concat({k: v for k, v in stock_daily_dict.items()}).reset_index(drop=True)
        #stock_df['frequency'] = 1  # 1天
        utils_data.output_database(stock_daily_df,
                                   filename='ods_ohlc_incr_efinance_stock_bj_daily')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2026-06-01', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='incr', help='Data update method')
    args = parser.parse_args()
    
    ods_efinance_stock_bj_api = odsEfinanceStockBjApi()
    date_end = args.date_end if args.date_end!='' else datetime.now().strftime("%F")
    if args.update_type=='full':
        ods_efinance_stock_bj_api.stock_daily(date_end=date_end)
    elif args.update_type=='incr':
        date_start = utils_data.maximum_date_next(table_name='ods_ohlc_incr_efinance_stock_bj_daily', field_name='日期')
        ods_efinance_stock_bj_api.stock_daily(date_start=date_start, date_end=date_end)


