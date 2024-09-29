# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:25:08 2024

@author: awei
北交所日频数据(data_ods_freq_incr_efinance_stock_bj_api)
一定要借助清洗后的dwd层的股票基本信息才能关联到到北京对应的股票代码
"""
import os

import pandas as pd
import efinance as ef # efinance不能连国际VPN 

from __init__ import path
from base import base_connect_database, base_log
from data import data_utils

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')


class odsEfinanceStockBjApi():
    def __init__(self):
        with base_connect_database.engine_conn('postgre') as conn:
            dwd_stock_base_df = pd.read_sql('dwd_info_incr_stock_base', con=conn.engine)
        self.dwd_stock_bj_base_df = dwd_stock_base_df[dwd_stock_base_df.ticker_suffix=='BJ']
        logger.info(f'dwd_stock_bj_base_df.shape: {self.dwd_stock_bj_base_df.shape}')
        
    def stock_bj_minute(self, frequency=5):
        # frequency in [5, 15, 30, 60, None]
        stock_bj_minute_dict = ef.stock.get_quote_history(self.dwd_stock_bj_base_df.code, klt=frequency) # 优先个股，其次ETF
        stock_bj_minute_df = pd.concat({k: v for k, v in stock_bj_minute_dict.items()}).reset_index(drop=True)
        #stock_bj_df['frequency'] = 5  # 5分钟
        data_utils.output_database(stock_bj_minute_df, 'ods_freq_incr_api_efinance_stock_bj_minute')
    
    def stock_bj_daily(self):
        stock_bj_daily_dict = ef.stock.get_quote_history(self.dwd_stock_bj_base_df.code) # 优先个股，其次ETF
        stock_bj_daily_df = pd.concat({k: v for k, v in stock_bj_daily_dict.items()}).reset_index(drop=True)
        #stock_bj_df['frequency'] = 1  # 1天
        data_utils.output_database(stock_bj_daily_df, 'ods_freq_incr_api_efinance_stock_bj_daily')
        
if __name__ == '__main__':
    ods_efinance_stock_bj_api = odsEfinanceStockBjApi()
    ods_efinance_stock_bj_api.stock_bj_daily()
    
    