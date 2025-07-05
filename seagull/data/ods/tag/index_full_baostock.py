# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取股票指数成分股信息(data_ods_part_baostock_index_api)

self.ods_full_baostock_index_api = ods_part_baostock_index_api.odsFullBaostockIndexApi()

elif data_type == '指数成分股':
self.ods_full_baostock_index_api.index_daily()
"""
import os
import argparse

import baostock as bs
import pandas as pd

from seagull.settings import PATH
from seagull.data import data_utils
from base import base_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{PATH}/log/{log_filename}.log')



class odsFullBaostockIndexApi():
    def __init__(self):
       super().__init__()
       
    def index(self, data_type):
        """
        获取常规数据
        :param data_type:获取数据类型
        备注：1.更新频率：天
        """
        logger.info(f'数据类型: {data_type}')
        bs.login()
        if data_type == '上证50':  # 上证50成分股
            rs = bs.query_sz50_stocks()
            index_df = data_utils.re_get_row_data(rs)
            index_df[['index','index_name']] = 'sz50', '上证50'
        elif data_type == '沪深300':  # 沪深300成分股
            rs = bs.query_hs300_stocks()
            index_df = data_utils.re_get_row_data(rs)
            index_df[['index','index_name']] = 'hs300', '沪深300'
        elif data_type == '中证500':  # 中证500成分股
            rs = bs.query_zz500_stocks()
            index_df = data_utils.re_get_row_data(rs)
            index_df[['index','index_name']] = 'zz500', '中证500'
        bs.logout()
        return index_df
        
    def index_daily(self):
        index_sz50_df = self.index(data_type='上证50')
        index_hs300_df = self.index(data_type='沪深300')
        index_zz500_df = self.index(data_type='中证500')
        index_df = pd.concat([index_sz50_df, index_hs300_df, index_zz500_df], axis=0)
        data_utils.output_database(index_df, 'ods_part_full_baostock_index', if_exists='replace')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='中证500', help='["上证50", "沪深300", "中证500"]')
    args = parser.parse_args()
    
    ods_full_baostock_index_api = odsFullBaostockIndexApi()
    #result = ods_api_part_baostock_index.index(args.data_type)
    #print(result)
    
    ods_full_baostock_index_api.index_daily()