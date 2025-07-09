# -*- coding: utf-8 -*-
"""
@Date: 2025/6/24 23:38
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_incr_daily.py
@Description:

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

ETF,创业板可以突破10%的限制和地域限制

"""
import pandas as pd

from seagull.utils import utils_database, utils_character, utils_log, utils_data, utils_thread


def dwd_freq_incr_portfolio_daily(self):
    # 清洗efinance的get_quote_history()接口数据
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql('ods_ohlc_fund_incr_efinance_daily', con=conn.engine)
        fund_base_df = pd.read_sql('ods_info_fund_full_efinance', con=conn.engine)
    fund_base_df = fund_base_df[['market_code', 'full_code', 'asset_code']]

    fund_df = fund_df.rename(columns={'股票名称': 'code_name',
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

    fund_df['freq_code'] = 101
    fund_df['adj_code'] = 1  # adjustment_type as adj_type in ['None', 'Pre', 'Post']
    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    fund_df['time'] = pd.to_datetime(fund_df['date']).dt.strftime("%Y%m%d%H%M%S")
    fund_df = pd.merge(fund_df, fund_base_df, on='asset_code')

    fund_df['primary_key'] = (fund_df['time'].astype(str) +
                              fund_df['full_code'].astype(str) +
                              fund_df['freq_code'].astype(str) +
                              fund_df['adj_code'].astype(str)
                              ).apply(utils_character.md5_str)  # md5（时间、代码、频率、复权）
    utils_data.output_database(fund_df, 'dwd_freq_incr_portfolio_daily')


def dwd_freq_full_portfolio_daily_backtest(self):
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql("dwd_freq_incr_portfolio_daily", con=conn.engine)
        # fund_df = pd.read_sql("select * from dwd_freq_incr_portfolio_daily where date between '2019-01-01' and '2023-01-01'", con=conn.engine)
    fund_df = fund_df[['date', 'full_code', 'close']]
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    portfolio_daily_backtest_df = fund_df.pivot(index='date', columns='full_code', values='close')
    utils_data.output_database(portfolio_daily_backtest_df, filename='dwd_freq_full_portfolio_daily_backtest',
                               index=True)
