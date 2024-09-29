# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取ods层基本信息(data_ods_info_incr_baostock_stock_base_api)
1交易日
2获取股票对应的板块
3获取证券基本资料
4获取证券代码
"""
import os
import argparse
import baostock as bs

from __init__ import path
from data import data_utils, data_ods_info_incr_baostock_trade_stock_api
from base import base_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')


class odsBaostockStockBaseApi:
    def __init__(self):
       super().__init__()
       
    def stock_base(self, data_type):
        """
        获取常规数据
        :param data_type:获取数据类型
        备注：1.更新频率：天
        """
        logger.info(f'数据类型: {data_type}')
        bs.login()
        if data_type == '交易日':
            rs = bs.query_trade_dates()
            filename = 'ods_info_incr_baostock_trade_dates'
            
        elif data_type == '行业分类':  # 获取行业分类数据
            rs = bs.query_stock_industry()
            filename = 'ods_info_full_baostock_stock_industry'
            
        elif data_type == '证券资料':  # 获取证券基本资料
            rs = bs.query_stock_basic()
            filename = 'ods_info_incr_baostock_stock_base'
        bs.logout()
        result = data_utils.re_get_row_data(rs)
        data_utils.output_database(result, filename)
        
        if data_type == '证券代码':
            #rs = bs.query_all_stock(date)
            date_start = data_utils.maximum_date('ods_info_incr_baostock_trade_stock')
            data_ods_info_incr_baostock_trade_stock_api.baostock_trade_stock(date_start=date_start)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='证券代码', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()
    
    ods_baostock_stock_base_api = odsBaostockStockBaseApi()
    ods_baostock_stock_base_api.stock_base(args.data_type)
    
    