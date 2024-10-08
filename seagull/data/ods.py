# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取ods层基本信息(data_ods)
1交易日
2获取股票对应的板块
3获取证券基本资料
4获取证券代码
"""
import argparse
from datetime import datetime, timedelta
from sqlalchemy import String  # Float, Numeric, 

import adata
import efinance as ef
import baostock as bs
import pandas as pd

from __init__ import path
from data import (data_loading,
                  ods_part_baostock_index_api)
from utils import utils_database, utils_log, utils_data


logger = utils_log.logger_config_local(f'{path}/log/data_ods.log')


class odsData():
    def __init__(self):
        #self.conn = utils_database.engine_conn('postgre')
        ods_full_baostock_index_api = ods_part_baostock_index_api.odsFullBaostockIndexApi()
        
    def ods_api_info_baostock(self, data_type):
        """
        获取常规数据
        :param data_type:获取数据类型
        备注：1.更新频率：天
        """
        logger.info(f'数据类型: {data_type}')
        if data_type == '交易日':
            rs = bs.query_trade_dates()
            filename = 'ods_api_info_baostock_trade_dates'
            
        elif data_type == '行业分类':  # 获取行业分类数据
            rs = bs.query_stock_industry()
            filename = 'ods_api_info_baostock_stock_industry'
            
        elif data_type == '证券资料':  # 获取证券基本资料
            rs = bs.query_stock_basic()
            filename = 'ods_api_info_baostock_allstocks_basic'
            
        elif data_type == '证券代码':
            #date = (datetime.now()+timedelta(days=-3)).strftime('%F')  # 只有交易日才会更新，取当天的不一定及时更新，先尝试前一天
            #rs = bs.query_all_stock(date)
            filename = 'ods_api_info_baostock_asset_base'
        
        elif data_type == '成分股':
            rs = bs.query_sz50_stocks()
            filename = 'ods_api_baostock_sz50_stocks'
            
            
        result = data_loading.re_get_row_data(rs)
        utils_data.output_database(result, filename)
        
    def ods_adata_portfolio_base(self, data_type):
        if data_type == 'ETF代码':
            result = adata.fund.info.all_etf_exchange_traded_info()
            filename = 'ods_info_nrtd_adata_portfolio_base'
            dtype={'primary_key': String,
                   'date': String,}
        utils_data.output_database(result, filename, dtype)
    
    def ods_efinance_portfolio(self, data_type):
        with utils_database.engine_conn('postgre') as conn:
            etf_code_df = pd.read_sql('ods_info_nrtd_adata_portfolio_base', con=conn.engine)
        if data_type == 'ETF日频':
            etf_dict = ef.stock.get_quote_history(etf_code_df.fund_code)
            filename = 'ods_freq_incr_efinance_portfolio_daily'
        elif data_type == 'ETF五分钟频':
            etf_dict = ef.stock.get_quote_history(etf_code_df.fund_code, klt=5)
            filename = 'ods_freq_incr_efinance_portfolio_minute'
        result = pd.concat({k: pd.DataFrame(v) for k, v in etf_dict.items()})
        utils_data.output_database(result, filename)
        
        
if __name__ == '__main__':
    lg = bs.login()  # 登陆系统
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='证券代码', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()
    
    ods_data = odsData()
    result = ods_data.api_baostock(args.data_type)
    print(result)
    
    bs.logout()  # 登出系统
    