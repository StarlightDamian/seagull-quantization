# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:29:50 2024

@author: awei
K线数据大宽表(data_dwd_stock_clean)
1.个股、ETF每日数据
baostock获取sz、sh的个股数据
efinance获取bj的个股数据
efinance获取etf数据
"""
import os
from sqlalchemy import Float, Numeric, String
# from datetime import datetime

import pandas as pd

from __init__ import path
from base import base_connect_database, base_utils, base_log
from data import data_utils, data_ods_api_freq_baostock_stock

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')


class dwdStockClean(data_ods_api_freq_baostock_stock.odsFreqShSzStock):
    def __init__(self):
        with base_connect_database.engine_conn('postgre') as conn:
            self.dwd_all_stock_df = pd.read_sql("dwd_info_incr_stock_base", con=conn.engine)# 获取指数、股票数据
            
    def clean_baostock_query_history_k_data_plus(self, data_raw_df):
        # 清洗baostock的query_history_k_data_plus()接口数据，用于生产上交所、深交所的股票时频数据
        data_raw_df = data_utils.split_baostock_code(data_raw_df)
        logger.info(data_raw_df[['date']].values[0][0])
        
        data_raw_df['full_code'] = data_raw_df['market_code'] + '.' + data_raw_df['asset_code']
        
        # logger.info(self.dwd_all_stock_df)
        data_raw_df = pd.merge(data_raw_df, self.dwd_all_stock_df[['full_code', 'code_name', 'board_type', 'price_limit_pct']], on='code_with_ticker_suffix')
        data_raw_df.loc[data_raw_df.isST=='1','price_limit_pct'] = 5
        
        # 对异常值补全. 部分'amount'、'volume'为''
        columns_float_list = ['open', 'high', 'low', 'close', 'preclose'] # 分钟线没有'preclose'
        data_raw_df[columns_float_list] = data_raw_df[columns_float_list].fillna(0).astype(float)
        
        # 更高级别的异常处理
        columns_apply_float_list = ['amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']  # 选择需要处理的字段列表，分钟线只有'amount'
        data_raw_df.loc[:, columns_apply_float_list] = data_raw_df[columns_apply_float_list].apply(pd.to_numeric, errors='coerce')  # 使用apply函数对每个字段进行处理
        data_raw_df[columns_apply_float_list] = data_raw_df[columns_apply_float_list].fillna(0).astype(float)  # 将 NaN 值填充为 0 或其他合适的值
        
        # volume中有异常值,太长无法使用.astype(int)。'adjustflag', 'tradestatus', 'isST',保持str
        data_raw_df.loc[:, ['volume']] = pd.to_numeric(data_raw_df['volume'], errors='coerce', downcast='integer')  # 使用pd.to_numeric进行转换，将错误的值替换为 NaN
        data_raw_df[['volume']] = data_raw_df[['volume']].fillna(0).astype('int64')  # 将 NaN 值填充为 0 或其他合适的值
        
        data_raw_df = data_raw_df.rename(columns={'pctChg': 'pct_chg',
                                                  'peTTM': 'pe_ttm',
                                                  'psTTM': 'ps_ttm',
                                                  'pcfNcfTTM': 'pcf_ncf_ttm',
                                                  'pbMRQ': 'pb_mrq',
                                                  'isST': 'is_st',
                                                  })
        
        # 没有时间的把日期转化为字符时间格式，方便后续统一主键
        data_raw_df['time'] = pd.to_datetime(data_raw_df['date']).dt.strftime('%Y%m%d%H%M%S')
        
        data_raw_df['adj_type'] = None   # adjustment_type as adj_type in ['None', 'Pre', 'Post']
        # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
        data_raw_df['primary_key'] = (data_raw_df['time'] +
                                      data_raw_df['full_code'] +
                                      data_raw_df['frequency'] +
                                      data_raw_df['adj_type']
                                      ).apply(base_utils.md5_str) # md5（时间、带后缀代码、频率）
        
        #data_raw_df[['primary_key', 'date','time', 'asset_code', 'open', 'high', 'low', 'close', 'volume', 'amount']] # 分钟级清洗保留字段
        data_raw_df = data_raw_df[['primary_key', 'date', 'time', 'board_type', 'market_code', 'asset_code', 'full_code', 'code_name', 'price_limit_pct', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pct_chg', 'pe_ttm', 'ps_ttm', 'pcf_ncf_ttm', 'pb_mrq', 'is_st','insert_timestamp']]
        logger.success('data handle')
        return data_raw_df
        
    def clean_efinance_get_quote_history(self, portfolio_daily_df):
        # 清洗efinance的get_quote_history()接口数据，用于生产全球股票数据
        portfolio_daily_df = portfolio_daily_df.rename(columns={'股票名称':'code_name',
                                                    '股票代码': 'asset_code',
                                                    '日期': 'date',
                                                    '开盘': 'open',
                                                    '收盘': 'close',
                                                    '最高': 'high',
                                                    '最低': 'low',
                                                    '成交量': 'volume',
                                                    '成交额': 'amount',
                                                    '振幅': 'amplitude',  # new
                                                    '涨跌幅': 'pct_chg',
                                                    '涨跌额': 'price_chg',  # new
                                                    '换手率': 'turn',
                                                    })
        portfolio_daily_df['adj_type'] = None   # adjustment_type as adj_type in ['None', 'Pre', 'Post']
        # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
        portfolio_daily_df['primary_key'] = (portfolio_daily_df['time'].astype(str) +
                                             portfolio_daily_df['full_code'].astype(str) +
                                             portfolio_daily_df['frequency'].astype(str) +
                                             portfolio_daily_df['adj_type'].astype(str)
                                             ).apply(base_utils.md5_str) # md5（时间、带后缀代码、频率）
        return portfolio_daily_df
    
    def dwd_stock_minute(self):
        with base_connect_database.engine_conn('postgre') as conn:
            baostock_stock_sh_sz_minute_df = pd.read_sql("ods_freq_incr_baostock_stock_sh_sz_minute", con=conn.engine)
        clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_minute_df)
        
        clean_asset_minute_df = pd.concat([clean_stock_sh_sz_df], axis=0)
        data_utils.output_database(clean_asset_minute_df,
                                   filename='dwd_freq_incr_stock_minute',
                                   dtype={'primary_key': String,
                                        'date': String,
                                        'time': String,
                                        'asset_code': String,
                                        'open': Float,
                                        'high': Float,
                                        'low': Float,
                                        'close': Float,
                                        'volume': Numeric,
                                        'amount': Numeric,
                                        })
        
    def dwd_stock_daily(self):
        with base_connect_database.engine_conn('postgre') as conn:
            baostock_stock_sh_sz_daily_df = pd.read_sql("ods_freq_incr_baostock_stock_sh_sz_daily", con=conn.engine)
            efinence_stock_bj_daily_df = pd.read_sql("ods_freq_incr_efinence_stock_bj_daily", con=conn.engine)
            efinance_portfolio_daily_df = pd.read_sql("ods_freq_incr_efinance_portfolio_daily", con=conn.engine)
        clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_daily_df)
        clean_stock_bj_df = self.clean_efinance_get_quote_history(efinence_stock_bj_daily_df)
        clean_portfolio_df = self.clean_efinance_get_quote_history(efinance_portfolio_daily_df)
        
        clean_asset_daily_df = pd.concat([clean_stock_sh_sz_df, clean_stock_bj_df, clean_portfolio_df], axis=0)
        data_utils.output_database(clean_asset_daily_df,  # 全量数据36分钟
                                   filename='dwd_freq_incr_stock_daily',
                                   dtype={'primary_key': String,
                                           'date': String,
                                           'market_code': String,
                                           'asset_code': String,
                                           'code_name': String,
                                           'open': Float,
                                           'high': Float,
                                           'low': Float,
                                           'close': Float,
                                           'preclose': Float,
                                           'volume': Numeric,
                                           'amount': Numeric,
                                           'amplitude': Float,
                                           'adjustflag': String,
                                           'turn': Float,
                                           'tradestatus': String,
                                           'pct_chg': Float,
                                           'price_chg': Float,
                                           'pe_ttm': Float,
                                           'ps_ttm': Float,
                                           'pcf_ncf_ttm': Float,
                                           'pb_mrq': Float,
                                           'is_st': String,
                                           })
        
if __name__ == '__main__':
    dwd_stock_clean = dwdStockClean()
    dwd_stock_clean.dwd_stock_daily()
    
# =============================================================================
#     a_stock_k_raw_df.columns
#     Out[2]: 
#     Index(['date', 'asset_code', 'open', 'high', 'low', 'close', 'preclose', 'volume',
#            'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
#            'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'insert_timestamp',
#            'primary_key'],
#           dtype='object')
# 
#     a_stock_k_df.columns
#     Out[3]: 
#     Index(['primary_key', 'date', 'asset_code', 'code_name', 'open', 'high', 'low','close', 'volume','amount','turn','pct_chg', 
#             'preclose',  'adjustflag', 
#            'tradestatus', 'pe_ttm', 'ps_ttm', 'pcf_ncf_ttm', 'pb_mrq',
#            'is_st','insert_timestamp'],
#           dtype='object')
# 
#     etf_day_df.columns
#     Out[4]: 
#     Index(['primary_key','code_name', 'asset_code', 'date', 'open', 'close', 'high', 'low', 'volume','amount','turn','pct_chg', 
#             'amplitude', 'price_chg', 
#            'insert_timestamp'],
#           dtype='object')
# =============================================================================
