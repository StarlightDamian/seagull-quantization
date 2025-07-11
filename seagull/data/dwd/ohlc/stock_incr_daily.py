# -*- coding: utf-8 -*-
"""
@Date: 2025/7/10 9:41
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_daily.py
@Description: 
"""

import os
import argparse
#from sqlalchemy.types import Boolean
import pandas as pd

from seagull.settings import PATH
from sqlalchemy import Float, Numeric, String  #, Integer

from seagull.utils import utils_database, utils_character, utils_log, utils_data, utils_thread
from seagull.utils.api import utils_api_baostock
from data.ods.ohlc import ods_ohlc_incr_baostock_stock_sh_sz_api
from finance import finance_limit
from finance.finance_trading_day import TradingDayAlignment
from feature import vwap, max_drawdown

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

def clean_baostock(self, data_raw_df):
    # 清洗baostock的query_history_k_data_plus()接口数据，用于生产上交所、深交所的股票时频数据
    print('1', data_raw_df.shape)
    data_raw_df = utils_api_baostock.split_baostock_code(data_raw_df)
    # logger.info(data_raw_df[['date']].values[0][0])
    print('2', data_raw_df.shape)
    data_raw_df['full_code'] = data_raw_df['asset_code'] + '.' + data_raw_df['market_code']
    print('3', data_raw_df.shape)
    # logger.info(self.dwd_all_stock_df) # debug
    global data_raw_df1, dwd_all_stock_df1
    data_raw_df1 = data_raw_df
    dwd_all_stock_df1 = self.dwd_all_stock_df
    data_raw_df = pd.merge(data_raw_df, self.dwd_all_stock_df, on='full_code', how='left')
    data_raw_df.loc[data_raw_df.isST == '1', ['board_type', 'price_limit_rate']] = 'ST', 0.05
    print('4', data_raw_df.shape)
    # 对异常值补全. 部分'amount'、'volume'为''
    columns_float_list = ['open', 'high', 'low', 'close', 'preclose']  # 分钟线没有'preclose'
    data_raw_df[columns_float_list] = data_raw_df[columns_float_list].fillna(0).astype(float)
    print('5', data_raw_df.shape)
    # 更高级别的异常处理
    columns_apply_float_list = ['amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM',
                                'pbMRQ']  # 选择需要处理的字段列表，分钟线只有'amount'
    data_raw_df.loc[:, columns_apply_float_list] = data_raw_df[columns_apply_float_list].apply(pd.to_numeric,
                                                                                               errors='coerce')  # 使用apply函数对每个字段进行处理
    data_raw_df[columns_apply_float_list] = data_raw_df[columns_apply_float_list].fillna(0).astype(
        float)  # 将 NaN 值填充为 0 或其他合适的值
    print('6', data_raw_df.shape)
    # volume中有异常值,太长无法使用.astype(int)。'adjustflag', 'tradestatus', 'isST',保持str
    data_raw_df.loc[:, ['volume']] = pd.to_numeric(data_raw_df['volume'], errors='coerce',
                                                   downcast='integer')  # 使用pd.to_numeric进行转换，将错误的值替换为 NaN
    data_raw_df[['volume']] = data_raw_df[['volume']].fillna(0).astype('int64')  # 将 NaN 值填充为 0 或其他合适的值
    print('7', data_raw_df.shape)
    data_raw_df = data_raw_df.rename(columns={'amount': 'turnover',
                                              'turn': 'turnover_pct',
                                              'preclose': 'prev_close',
                                              'tradestatus': 'is_trade',
                                              'pctChg': 'chg_rel',
                                              'peTTM': 'pe_ttm',
                                              'psTTM': 'ps_ttm',
                                              'pcfNcfTTM': 'pcf_ttm',
                                              'pbMRQ': 'pb_mrq',
                                              'isST': 'is_st',
                                              })
    data_raw_df['is_st'] = data_raw_df['is_st'].map({'1': True,
                                                     '0': False})
    data_raw_df['is_trade'] = data_raw_df['is_trade'].map({'1': True,
                                                           '0': False})
    print('8', data_raw_df.shape)
    data_raw_df = self.clean_general(data_raw_df)
    print('9', data_raw_df.shape)
    # data_raw_df[['primary_key', 'date','time', 'asset_code', 'open', 'high', 'low', 'close', 'volume', 'amount']] # 分钟级清洗保留字段
    data_raw_df = data_raw_df[
        ['primary_key', 'adj_type', 'freq', 'date', 'time', 'board_type', 'full_code', 'asset_code', 'market_code',
         'code_name', 'price_limit_rate', 'open', 'high', 'low', 'close', 'prev_close', 'volume', 'turnover',
         'turnover_pct', 'is_trade', 'chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq', 'is_st', 'insert_timestamp']]
    logger.success('data handle')
    print('10', data_raw_df.shape)
    return data_raw_df

    def clean_efinance_get_quote_history(self, portfolio_daily_df):
        # 清洗efinance的get_quote_history()接口数据，用于生产全球股票数据
        portfolio_daily_df = portfolio_daily_df.rename(columns={'股票名称': 'code_name',
                                                                '股票代码': 'asset_code',
                                                                '日期': 'date',
                                                                '开盘': 'open',
                                                                '收盘': 'close',
                                                                '最高': 'high',
                                                                '最低': 'low',
                                                                '成交量': 'volume',
                                                                '成交额': 'amount',
                                                                '振幅': 'amplitude',  # new
                                                                '涨跌幅': 'chg_rel',
                                                                '涨跌额': 'price_chg',  # new
                                                                '换手率': 'turnover',
                                                                })

        portfolio_daily_df['full_code'] = portfolio_daily_df['asset_code'] + '.' + portfolio_daily_df['market_code']
        portfolio_daily_df['is_st'] = False
        portfolio_daily_df['is_trade'] = True

        portfolio_daily_df = self.clean_general(portfolio_daily_df)

        return portfolio_daily_df

def dwd_ohlc_stock_incr_daily(self, df):
    df['freq'] = self.freq
    df['adj_type'] = self.adj_type  # adjustment_type as adj_type in ['None', 'pre', 'post']

    # 没有时间的把日期转化为字符时间格式，方便后续统一主键
    df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')

    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    df['primary_key'] = (df['time'].astype(str) +
                         df['full_code'].astype(str) +
                         df['freq'].astype(str) +
                         df['adj_type'].astype(str)
                         ).apply(utils_character.md5_str)  # md5（时间、带后缀代码、频率、复权）
    return df


def pipeline():
    stock_df = dwd_ohlc_stock_incr_daily()
    utils_data.output_database_large(stock_df,
                                     filename='dwd_ohlc_stock_incr_daily',
                                     if_exists='replace')


if __name__ == '__main__':
    pipeline()
