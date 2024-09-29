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
from data import data_loading
from base import base_connect_database, base_log


logger = base_log.logger_config_local(f'{path}/log/data_ods.log')


class odsData():
    def __init__(self):
        self.conn = base_connect_database.engine_conn('postgre')
        

        
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
            date = (datetime.now()+timedelta(days=-3)).strftime('%F')  # 只有交易日才会更新，取当天的不一定及时更新，先尝试前一天
            rs = bs.query_all_stock(date)
            filename = 'ods_api_info_baostock_asset_base'
        
        elif data_type == '上证50':  # 上证50成分股
            rs = bs.query_sz50_stocks()
            filename = 'ods_api_baostock_sz50_stocks'
            
        elif data_type == '沪深300':  # 沪深300成分股
            rs = bs.query_hs300_stocks()
            filename = 'ods_api_baostock_hs300_stocks'
            
        elif data_type == '中证500':  # 中证500成分股
            rs = bs.query_zz500_stocks()
            filename = 'ods_api_baostock_zz500_stocks'
            
        result = data_loading.re_get_row_data(rs)
        self.output_database(result, filename)
        
    def ods_api_info_adata_etf_code(self, data_type):
        if data_type == 'ETF代码':
            result = adata.fund.info.all_etf_exchange_traded_info()
            filename = 'ods_api_info_adata_etf_code'
            dtype={'primary_key': String,
                  'date': String,}
        self.output_database(result, filename, dtype)
    
    def ods_api_freq_efinance_etf_daily(self, data_type):
        etf_code_df = pd.read_sql('ods_api_info_adata_etf_code', con=self.conn.engine)
        if data_type == 'ETF日频':
            etf_dict = ef.stock.get_quote_history(etf_code_df.fund_code)
            filename = 'ods_api_freq_efinance_etf_daily'
        elif data_type == 'ETF五分钟频':
            etf_dict = ef.stock.get_quote_history(etf_code_df.fund_code, klt=5)
            filename = 'ods_api_freq_efinance_etf_5min'
        result = pd.concat({k: pd.DataFrame(v) for k, v in etf_dict.items()})
        self.output_database(result, filename)
        
        
if __name__ == '__main__':
    lg = bs.login()  # 登陆系统
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='证券代码', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()
    
    ods_data = odsData()
    result = ods_data.api_baostock(args.data_type)
    print(result)
    
    bs.logout()  # 登出系统
    