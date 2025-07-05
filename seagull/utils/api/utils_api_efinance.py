# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:52:59 2024

@author: awei
efinance的api数据处理工具(utils_api_efinance)
1. efinanc不能翻墙
1. efinance会获取当日数据，所以要剔除当日数据
"""
from datetime import datetime

import efinance as ef
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database


def efinance_codelist2stock(code_arr=None):
    if not code_arr:
        with utils_database.engine_conn("POSTGRES") as conn:
            etf_code_df = pd.read_sql('ods_info_nrtd_adata_portfolio_base', con=conn.engine)
        code_arr = etf_code_df.fund_code
    etf_dict = ef.stock.get_quote_history(code_arr)
    stock_df = pd.concat({k: pd.DataFrame(v) for k, v in etf_dict.items()})
    
    today = datetime.now().strftime("%F")
    stock_df = stock_df[~(stock_df['日期']==today)]
    
    return stock_df.reset_index(drop=True)

import efinance as ef
if __name__ == '__main__': 
    # code_arr = ['HSI','SPX']
    #
    # index_dict = {'HSI': '恒生指数',
    #               'HSTECH': '恒生科技指数',
    #               'TQQQ': '三倍做多纳斯达克100ETF',
    #               'HXC': '纳斯达克中国金龙指数',
    #               'SPX': '标准普尔500指数',
    #               'YANG': '三倍做空富时中国ETF-Direxion',
    #               }
    # df = efinance_codelist2stock(code_arr)


    # 股票代码
    stock_code = 'AAPL'
    df = ef.stock.get_quote_history(stock_code, klt=5)
