# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:00:15 2023

@author: awei
获取指定日期全部股票的日K线数据(data_ods_freq_incr_baostock_stock_sh_sz_api)
code_name 不属于特征，在这一层加入
5、15、30、60分钟线指标参数(不包含指数)

"""
import os

import baostock as bs
import pandas as pd

from __init__ import path
from base import base_connect_database, base_log
from data import data_utils

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')


class odsIncrBaostockStockShSzApi():
    """
    A股的K线数据，全量历史数据接口
    """
    def __init__(self):
        with base_connect_database.engine_conn('postgre') as conn:
            self.ods_stock_base_df = pd.read_sql("ods_info_incr_baostock_stock_base", con=conn.engine)# 获取指数、股票数据
        
    def stock_sh_sz_1(self, substring_pd,
                            date_start,
                            date_end,
                            fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
                            frequency='d',
                            adjustflag='3'):
        code = substring_pd.name
        logger.debug(f'code: {code}| date_start: {self.date_start}| date_end: {self.date_end}')
        k_rs = bs.query_history_k_data_plus(code,
                                            fields=fields,
                                            date_start=date_start,
                                            date_end=date_end,
                                            frequency=frequency,
                                            adjustflag=adjustflag
                                            )
        try:
            data_df = k_rs.get_data()
        except:
            logger.error(code)
        if data_df.empty:
            logger.debug(f'{code} empty')
        else:
            logger.debug(f'{code} {data_df.shape}')
            return data_df
        
    def stock_sh_sz(self, date_start='1990-01-01',
                          date_end='2100-01-01',  
                          fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST',
                          frequency='d',
                          adjustflag='3'):
        bs.login()
        data_df = self.ods_stock_base_df.groupby('code').apply(self.stock_sh_sz_1, fields=fields, frequency=frequency, adjustflag=adjustflag)     
        bs.logout()
        return data_df.reset_index(drop=True) 
    

    
