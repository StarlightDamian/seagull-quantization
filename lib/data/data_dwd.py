# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取dwd层基本信息(data_dwd)
1股票基本信息
2股票标签

Now I have three considerations
1. I want to use a code abbreviation across various assets, including stocks, real estate, bonds, futures, and ETFs
2. My database table contains fields such as 'SH', '510300', and 'SH.510300'. I hope to give them three columns with corresponding field names.
3. This naming is best for China and the United States

For Chinese ETF:
    market_code: 'SH'
    asset_code: '510300'
    full_code: 'SH.510300'


For US Stock:
    market_code: 'NYSE'
    asset_code: 'AAPL'
    full_code: 'NYSE.AAPL'
    
1.market_code: Represents the market or exchange (e.g., 'SH' for Shanghai, 'SZ' for Shenzhen, 'NYSE' for New York Stock Exchange)
2.asset_code: The specific identifier for the asset (e.g., '510300' for an ETF in China, or 'AAPL' for Apple stock)
3.full_code: A combination of market_code and asset_code (e.g., 'SH.510300' or 'NYSE.AAPL')
4.asset_type: Specifies the type of asset (e.g., 'STOCK', 'ETF', 'BOND', 'FUTURE', 'REIT')
5.asset_name: The full name of the asset
"""
import argparse
from loguru import logger
from datetime import datetime

import pandas as pd

from __init__ import path
from base import base_connect_database, base_utils, base_data

#logger.add(f"{path}/log/data_plate.log", rotation="10 MB", retention="10 days")

class dwdData():
    def __init__(self):
        ...
    
    def dwd_freq_full_portfolio_daily_backtest(self):
        with base_connect_database.engine_conn('postgre') as conn:
            portfolio_daily_df = pd.read_sql("dwd_freq_incr_portfolio_daily", con=conn.engine)
            #portfolio_daily_df = pd.read_sql("select * from dwd_freq_incr_portfolio_daily where date between '2019-01-01' and '2023-01-01'", con=conn.engine)
        portfolio_daily_df = portfolio_daily_df[['date','full_code', 'close']]
        portfolio_daily_df['date'] = pd.to_datetime(portfolio_daily_df['date'])
        portfolio_daily_backtest_df = portfolio_daily_df.pivot(index='date', columns='full_code', values='close')
        base_data.output_database(portfolio_daily_backtest_df, filename='dwd_freq_full_portfolio_daily_backtest', index=True)
    
    def full(self):
        self.dwd_freq_full_portfolio_daily_backtest()
    
    def dwd_info_nrtd_portfolio_base(self):
        with base_connect_database.engine_conn('postgre') as conn:
            portfolio_base_df = pd.read_sql('ods_info_nrtd_adata_portfolio_base', con=conn.engine)
            
        portfolio_base_df = portfolio_base_df.rename(columns={'fund_code': 'asset_code',
                                                                'short_name': 'code_name',
                                                                'net_value': 'prev_close',
                                                                })
        portfolio_base_df.market_code = portfolio_base_df.market_code.map({1: 'SH',
                                                                           0: 'SZ',
                                                                           })
        portfolio_base_df['full_code'] = portfolio_base_df.market_code + '.' + portfolio_base_df.asset_code
        base_data.output_database(portfolio_base_df, filename='dwd_info_nrtd_portfolio_base', if_exists='replace')
        
    def dwd_info_nrtd_bond_base(self):
        with base_connect_database.engine_conn('postgre') as conn:
            bond_base_df = pd.read_sql('ods_info_nrtd_adata_bond_base', con=conn.engine)
            
            bond_base_df = bond_base_df.rename(columns={'bond_code': 'bond_asset_code',
                                                                    'bond_name': 'bond_code_name',
                                                                    'stock_code': 'stock_asset_code',
                                                                    'short_name':'stock_code_name',
                                                                    #'sub_date'
                                                                    #'issue_amount'
                                                                    #'listing_date'
                                                                    #'expire_date'
                                                                    'convert_price': 'prev_close',
                                                                    'market_id': 'market_code',
                                                                    'stock_market_id': 'stock_market_id',
                                                                    })

        bond_base_df['market_code'] = bond_base_df['market_code'].map({'35': 'SZ',
                                                   '19': 'SH'})
        bond_base_df['stock_market_code'] = bond_base_df['stock_market_code'].map({'35': 'SZ',
                                                               '19': 'SH'})
        bond_base_df['stock_full_code'] = bond_base_df.stock_market_id + '.' + bond_base_df.stock_asset_code
        bond_base_df['bond_full_code'] = bond_base_df.bond_market_id + '.' + bond_base_df.bond_asset_code
        base_data.output_database(bond_base_df, 'dwd_info_nrtd_bond_base')
        
    def nrtd(self):
        self.dwd_info_nrtd_portfolio_base()
        self.dwd_info_nrtd_bond_base()

    def dwd_freq_incr_portfolio_daily(self):
        # 清洗efinance的get_quote_history()接口数据，用于生产全球股票数据
        with base_connect_database.engine_conn('postgre') as conn:
            portfolio_daily_df = pd.read_sql('ods_freq_incr_efinance_portfolio_daily', con=conn.engine)
            portfolio_base_df = pd.read_sql('dwd_info_nrtd_portfolio_base', con=conn.engine)
        portfolio_base_df = portfolio_base_df[['market_code','full_code','asset_code']]
        
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
        portfolio_daily_df['frequency'] = 'd'
        # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
        portfolio_daily_df['time'] = pd.to_datetime(portfolio_daily_df['date']).dt.strftime("%Y%m%d%H%M%S")
        portfolio_daily_df = pd.merge(portfolio_daily_df, portfolio_base_df, on='asset_code')

        portfolio_daily_df['primary_key'] = (portfolio_daily_df['time'].astype(str) +
                                             portfolio_daily_df['full_code'].astype(str) +
                                             portfolio_daily_df['frequency'].astype(str) +
                                             portfolio_daily_df['adj_type'].astype(str)
                                             ).apply(base_utils.md5_str) # md5（时间、带后缀代码、频率）
        base_data.output_database(portfolio_daily_df, 'dwd_freq_incr_portfolio_daily')
        
        
    def incr(self):
        self.dwd_freq_incr_portfolio_daily()
        
    def real(self):
        ...
        
    def pipline(self, data_type):
        """
        获取常规数据
        :param data_type:获取数据类型
        备注：1.更新频率：天
        """
        logger.info(f'数据类型: {data_type}')
        if data_type == '证券代码':
            result = data_dwd_all_stock.dwd_all_stock()
            filename = 'dwd_allstocks_basic'
            
        elif data_type == '证券标签':
            with base_connect_database.engine_conn('postgre') as conn:
                result = pd.read_sql('ods_api_baostock_stock_industry', con=conn.engine)
            result = result.rename(columns={'tradeStatus': 'trade_status'})
            filename = 'dwd_stock_label'
            
        base_data.output_database(result, filename)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='证券代码', help='["证券代码", "证券标签"]')
    args = parser.parse_args()
    
    dwd_data = dwdData()
    result = dwd_data.pipline(args.data_type)
    print(result)
    
