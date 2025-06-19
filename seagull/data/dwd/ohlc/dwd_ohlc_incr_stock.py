# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:29:50 2024

@author: awei
K线数据大宽表(dwd_ohlc_incr_stock)
1.个股、ETF每日数据
baostock获取sz、sh的个股数据
efinance获取bj的个股数据
efinance获取etf数据

trade_status
10d_max_recovery
"""

import os
import argparse
#from sqlalchemy.types import Boolean
import pandas as pd

from __init__ import path
from sqlalchemy import Float, Numeric, String  #, Integer

from utils import utils_database, utils_character, utils_log, utils_data, utils_thread
from data import utils_api_baostock
from data.ods.ohlc import ods_ohlc_incr_baostock_stock_sh_sz_api
from finance import finance_limit
from finance.finance_trading_day import TradingDayAlignment
from feature import vwap, max_drawdown

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

def bar_high(raw_df, window=10):
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['high','close']]
    df['y_high'] = df['high'].rolling(window=window, min_periods=1).max()
    df['y_high'] = df['y_high'].shift(-window+1)
    raw_df['y_high_rate'] = df['y_high'].div(df['close'], axis=0)
    return raw_df

def bar_low(raw_df, window=10):
    raw_df = raw_df.sort_values(by='date')
    df = raw_df[['low','close']]
    df['y_low'] = df['low'].rolling(window=window, min_periods=1).min()
    df['y_low'] = df['y_low'].shift(-window+1)
    raw_df['y_low_rate'] = df['y_low'].div(df['close'], axis=0)
    return raw_df

def n_day_pred(raw_df, window=10, freq='d'):
    freq = f'{window}{freq}'
    
    # N日平均成本
    raw_df = raw_df.groupby('full_code').apply(vwap.daily_vwap, window=window)
    raw_df = raw_df.rename(columns={'y_vwap_rate': f'y_{window}d_vwap_rate'}).reset_index(drop=True)
    
    # N日最大回撤
    raw_df = raw_df.groupby('full_code').apply(max_drawdown.calculate_max_drawdown, window=window)
    raw_df = raw_df.rename(columns={'y_max_dd': f'y_{window}d_max_dd'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','y_10d_max_dd']]
    
    # N日最大反弹
    raw_df = raw_df.groupby('full_code').apply(max_drawdown.calculate_max_recovery, window=window)
    raw_df = raw_df.rename(columns={'y_max_recovery': f'y_{window}d_max_recovery'}).reset_index(drop=True)
    
    # N日最高价
    raw_df = raw_df.groupby('full_code').apply(bar_high, window=window)
    raw_df = raw_df.rename(columns={'y_high_rate': f'y_{window}d_high_rate'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','high','y_10d_high','y_10d_high_rate']]
    
    # N日最低价
    raw_df = raw_df.groupby('full_code').apply(bar_low, window=window)
    raw_df = raw_df.rename(columns={'y_low_rate': f'y_{window}d_low_rate'}).reset_index(drop=True)
    # raw_df.tail(11)[['close','low','y_10d_low','y_10d_low_rate']]
    
    # VWAP-to-Drawdown Ratio
    raw_df[f'y_{window}d_vwap_drawdown_rate'] = raw_df[f'y_{window}d_vwap_rate'] / (raw_df[f'y_{window}d_max_dd'] + 1)
    return raw_df

def calculate_prev_close(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    df[['prev_close']] = df[['close']].shift(1)
    return df

class DwdStock(ods_ohlc_incr_baostock_stock_sh_sz_api.OdsIncrBaostockStockShSzApi):#OdsFreqShSzStock
    def __init__(self, freq, adj_type):
        self.freq=freq
        self.adj_type = adj_type  # adjustment_type as adj_type in ['None', 'pre', 'post']
        
        self.trading_day_alignment = TradingDayAlignment()
        with utils_database.engine_conn('postgre') as conn:
            dwd_all_stock_df = pd.read_sql("select full_code, code_name, board_type, price_limit_rate from dwd_info_incr_stock_base", con=conn.engine)  # 获取指数、股票数据
            self.dwd_all_stock_df = dwd_all_stock_df.drop_duplicates('full_code', keep='first')
            
    def clean_general(self, df):
        df['freq'] = self.freq
        df['adj_type'] = self.adj_type   # adjustment_type as adj_type in ['None', 'pre', 'post']
        
        # 没有时间的把日期转化为字符时间格式，方便后续统一主键
        df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
        
        # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
        df['primary_key'] = (df['time'].astype(str) +
                            df['full_code'].astype(str) +
                            df['freq'].astype(str) +
                            df['adj_type'].astype(str)
                            ).apply(utils_character.md5_str) # md5（时间、带后缀代码、频率、复权）
        return df
    
    def clean_baostock_query_history_k_data_plus(self, data_raw_df):
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
        data_raw_df.loc[data_raw_df.isST=='1', ['board_type', 'price_limit_rate']] = 'ST', 0.05
        print('4', data_raw_df.shape)
        # 对异常值补全. 部分'amount'、'volume'为''
        columns_float_list = ['open', 'high', 'low', 'close', 'preclose'] # 分钟线没有'preclose'
        data_raw_df[columns_float_list] = data_raw_df[columns_float_list].fillna(0).astype(float)
        print('5', data_raw_df.shape)
        # 更高级别的异常处理
        columns_apply_float_list = ['amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']  # 选择需要处理的字段列表，分钟线只有'amount'
        data_raw_df.loc[:, columns_apply_float_list] = data_raw_df[columns_apply_float_list].apply(pd.to_numeric, errors='coerce')  # 使用apply函数对每个字段进行处理
        data_raw_df[columns_apply_float_list] = data_raw_df[columns_apply_float_list].fillna(0).astype(float)  # 将 NaN 值填充为 0 或其他合适的值
        print('6', data_raw_df.shape)
        # volume中有异常值,太长无法使用.astype(int)。'adjustflag', 'tradestatus', 'isST',保持str
        data_raw_df.loc[:, ['volume']] = pd.to_numeric(data_raw_df['volume'], errors='coerce', downcast='integer')  # 使用pd.to_numeric进行转换，将错误的值替换为 NaN
        data_raw_df[['volume']] = data_raw_df[['volume']].fillna(0).astype('int64')  # 将 NaN 值填充为 0 或其他合适的值
        print('7', data_raw_df.shape)
        data_raw_df = data_raw_df.rename(columns={'amount': 'turnover',
                                                  'turn':'turnover_pct',
                                                  'preclose': 'prev_close',
                                                  'tradestatus': 'is_trade',
                                                  'pctChg': 'chg_rel',
                                                  'peTTM': 'pe_ttm',
                                                  'psTTM': 'ps_ttm',
                                                  'pcfNcfTTM': 'pcf_ttm',
                                                  'pbMRQ': 'pb_mrq',
                                                  'isST': 'is_st',
                                                  })
        data_raw_df['is_st'] = data_raw_df['is_st'].map({'1':True,
                                                         '0':False})
        data_raw_df['is_trade'] = data_raw_df['is_trade'].map({'1':True,
                                                               '0':False})
        print('8', data_raw_df.shape)
        data_raw_df = self.clean_general(data_raw_df)
        print('9', data_raw_df.shape)
        #data_raw_df[['primary_key', 'date','time', 'asset_code', 'open', 'high', 'low', 'close', 'volume', 'amount']] # 分钟级清洗保留字段
        data_raw_df = data_raw_df[['primary_key','adj_type', 'freq', 'date', 'time', 'board_type', 'full_code', 'asset_code', 'market_code', 'code_name', 'price_limit_rate', 'open', 'high', 'low', 'close', 'prev_close', 'volume', 'turnover', 'turnover_pct', 'is_trade', 'chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq', 'is_st', 'insert_timestamp']]
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
    
    def stock_minute(self):
        with utils_database.engine_conn('postgre') as conn:
            baostock_stock_sh_sz_minute_df = pd.read_sql("ods_ohlc_incr_baostock_stock_sh_sz_minute", con=conn.engine)
        clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_minute_df)
        
        clean_asset_minute_df = pd.concat([clean_stock_sh_sz_df], axis=0)
        utils_data.output_database(clean_asset_minute_df,
                                   filename='dwd_ohlc_incr_stock_minute',
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

    def stock_daily_full(self):
        with utils_database.engine_conn('postgre') as conn:
            # 沪深
            sql=f"""
                select
                    *
                from
                    ods_ohlc_incr_baostock_stock_sh_sz_daily 
                where
                    freq='{self.freq}'
                    and adj_type='{self.adj_type}'
                 """
            logger.info(sql)
            baostock_stock_sh_sz_daily_df = pd.read_sql(sql, con=conn.engine)
            
            # 北交所
            efinance_stock_bj_daily_df = pd.read_sql("ods_ohlc_incr_efinance_stock_bj_daily", con=conn.engine)
            efinance_stock_bj_daily_df['market_code'] = 'bj'
            efinance_stock_bj_daily_df[['board_type', 'price_limit_rate']] = '北交所', 0.3
            
            # ETF
            efinance_portfolio_daily_df = pd.read_sql("ods_ohlc_incr_efinance_portfolio_daily", con=conn.engine)
            dwd_portfolio_base_df = pd.read_sql("dwd_info_nrtd_portfolio_base", con=conn.engine)
            portfolio_market_dict = dict(zip(dwd_portfolio_base_df['asset_code'], dwd_portfolio_base_df['market_code']))
            efinance_portfolio_daily_df['market_code'] = efinance_portfolio_daily_df['股票代码'].map(portfolio_market_dict)
            efinance_portfolio_daily_df[['board_type', 'price_limit_rate']] = 'ETF', 0.1
            efinance_portfolio_daily_df.loc[efinance_portfolio_daily_df['股票名称'].str.contains('科创|创业|双创'), 'price_limit_rate'] = 0.2
            
        clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_daily_df)
        clean_stock_bj_df = self.clean_efinance_get_quote_history(efinance_stock_bj_daily_df)
        clean_portfolio_df = self.clean_efinance_get_quote_history(efinance_portfolio_daily_df)
        print('2', clean_stock_sh_sz_df.shape)
        print('2bj', clean_stock_bj_df.shape)
        print('2etf', clean_portfolio_df.shape)
        #数据来源表
        clean_stock_sh_sz_df['source_table'] = 'ods_ohlc_incr_baostock_stock_sh_sz_daily'
        clean_stock_bj_df['source_table'] = 'ods_ohlc_incr_efinance_stock_bj_daily'
        clean_portfolio_df['source_table'] = 'ods_ohlc_incr_efinance_portfolio_daily'
        
        asset_daily_df = pd.concat([clean_stock_sh_sz_df, clean_stock_bj_df, clean_portfolio_df], axis=0)
        #return asset_daily_df, clean_stock_sh_sz_df, clean_stock_bj_df, clean_portfolio_df
        print('3', asset_daily_df.shape)
        # zero_volume = asset_daily_df['volume']==0
        # asset_daily_df.loc[zero_volume, 'avg_price'] = np.nan
        # asset_daily_df.loc[~zero_volume,'avg_price'] = asset_daily_df.loc[~zero_volume, 'turnover'] / asset_daily_df.loc[~zero_volume, 'volume']#每单位成交量的平均价格
        # asset_daily_df['avg_price'] = asset_daily_df['avg_price'].round(4)
        
        # 使用 ThreadPoolExecutor 并行处理
        grouped = asset_daily_df.groupby('full_code')
        asset_daily_df = utils_thread.thread(grouped, calculate_prev_close, max_workers=8)
        print('4', asset_daily_df.shape)
        grouped = asset_daily_df.groupby('full_code') # 是两个grouped，否则上一个计算的结果不会输出
        asset_daily_df = utils_thread.thread(grouped, finance_limit.limit_prices, max_workers=8)
        print('5', asset_daily_df.shape)
        # KeyError: 'full_code'
        
        #global asset_daily_df1
        #asset_daily_df1 = asset_daily_df
        asset_daily_df['volume'] = asset_daily_df['volume'].astype('int64')
        asset_daily_df = n_day_pred(asset_daily_df, window=10, freq='d')
        print('6', asset_daily_df.shape)
        ## date
        trading_day_prev_dict = self.trading_day_alignment.prev_trading_day_dict(date_num=1)
        asset_daily_df['prev_date'] = asset_daily_df.date.map(trading_day_prev_dict)
        print('7', asset_daily_df.shape)
        trading_day_next_dict = self.trading_day_alignment.prev_trading_day_dict(date_num=-1)
        asset_daily_df['next_date'] = asset_daily_df.date.map(trading_day_next_dict)
        print('8', asset_daily_df.shape)
        # 特征: 日期_距离上一次开盘天数
        asset_daily_df['date_diff_prev'] = (pd.to_datetime(asset_daily_df.prev_date) - pd.to_datetime(asset_daily_df.date)).dt.days
        asset_daily_df['date_diff_prev'] =  asset_daily_df['date_diff_prev'].fillna(-1)
        asset_daily_df['date_diff_next'] = (pd.to_datetime(asset_daily_df.next_date) - pd.to_datetime(asset_daily_df.date)).dt.days
        asset_daily_df['date_diff_next'] =  asset_daily_df['date_diff_next'].fillna(1)
        print('9', asset_daily_df.shape)
        # 特征：日期_星期
        asset_daily_df['date_week'] = pd.to_datetime(asset_daily_df['date'], format='%Y-%m-%d').dt.day_name()
        print('11', asset_daily_df.shape)
        # asset_daily_df = asset_daily_df.groupby('full_code').apply(prev_close)
        # asset_daily_df = asset_daily_df.groupby('full_code').apply(finance_limit.limit_prices)
        # asset_daily_df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # 比较合理的拼接是：国家+股票市场+限价百分比, 用板块+限价百分比比较快速,后面可能加个股票性质是高股息还是妖股、蓝筹股
        asset_daily_df['board_primary_key'] = (asset_daily_df['board_type'].astype(str) +
                                                     asset_daily_df['price_limit_rate'].astype(str)
                                                     # asset_daily_df['is_limit_up_prev'].astype(str) +
                                                     # asset_daily_df['is_limit_down_prev'].astype(str) +
                                                     # asset_daily_df['is_flat_price_prev'].astype(str)
                                                     ).apply(utils_character.md5_str)
        print('12', asset_daily_df.shape)
        float_columns=['date_diff_prev','date_diff_next','price_limit_rate','prev_close','volume','turnover','amplitude',
                       'turnover_pct','chg_rel','price_chg','pe_ttm','ps_ttm','pcf_ttm','pb_mrq',
                       'limit_up','limit_down']#,'avg_price'
        asset_daily_df[float_columns] = asset_daily_df[float_columns].fillna(0)
        print('13', asset_daily_df.shape)
        # 全量数据36分钟跑完
        asset_daily_df = asset_daily_df.astype({'primary_key': 'string',
                                                'board_primary_key': 'string',
                                                'date': 'string',
                                                'prev_date': 'string',
                                                'date_diff_prev': 'float',
                                                'date_diff_next': 'float',
                                                'date_week': 'string',
                                                'time': 'string',
                                                'freq': 'string',
                                                'board_type': 'string',
                                                'price_limit_rate': 'float',
                                                'full_code': 'string',
                                                'asset_code': 'string',
                                                'market_code': 'string',
                                                'code_name': 'string',
                                                'open': 'float',
                                                'high': 'float',
                                                'low': 'float',
                                                'close': 'float',
                                                'prev_close': 'float',
                                                'volume': 'int64',
                                                'turnover': 'float',
                                                'amplitude': 'float',
                                                'adj_type': 'string',
                                                'turnover_pct': 'float',
                                                'chg_rel': 'float',
                                                'price_chg': 'float',
                                                'pe_ttm': 'float',
                                                'ps_ttm': 'float',
                                                'pcf_ttm': 'float',
                                                'pb_mrq': 'float',
                                                'source_table': 'string',
                                                'limit_up': 'float',
                                                'limit_down': 'float',
                                                #'avg_price': 'float',
                                                'is_st': 'bool',
                                                'is_trade': 'bool',
                                                'is_limit_up_prev': 'bool',
                                                'is_limit_down_prev': 'bool',
                                                #'is_flat_price_prev': 'bool',
                                                })
        print('14', asset_daily_df.shape)
        return asset_daily_df
# =============================================================================
#         utils_data.output_database_large(asset_daily_df,
#                                          filename='dwd_ohlc_incr_stock_daily',
#                                          if_exists='replace',
#                                          )
# =============================================================================
        
    def stock_daily_incr(self, df, date_start):
        prev_date = self.trading_day_alignment.shift_day(date_num=1)
        date = (prev_date, date_start)
        with utils_database.engine_conn('postgre') as conn:
            # 沪深
            sql=f"""
                select
                    *
                from
                    ods_ohlc_incr_baostock_stock_sh_sz_daily 
                where
                    freq='{self.freq}'
                    and adj_type='{self.adj_type}'
                    and date_start in {date}
                 """
            logger.info(sql)
            baostock_stock_sh_sz_daily_df = pd.read_sql(sql, con=conn.engine)
            
            # 北交所
            efinance_stock_bj_daily_df = pd.read_sql("ods_ohlc_incr_efinance_stock_bj_daily", con=conn.engine)
            efinance_stock_bj_daily_df['market_code'] = 'bj'
            efinance_stock_bj_daily_df[['board_type', 'price_limit_rate']] = '北交所', 0.3
            
            # ETF
            efinance_portfolio_daily_df = pd.read_sql("ods_ohlc_incr_efinance_portfolio_daily", con=conn.engine)
            dwd_portfolio_base_df = pd.read_sql("dwd_info_nrtd_portfolio_base", con=conn.engine)
            portfolio_market_dict = dict(zip(dwd_portfolio_base_df['asset_code'], dwd_portfolio_base_df['market_code']))
            efinance_portfolio_daily_df['market_code'] = efinance_portfolio_daily_df['股票代码'].map(portfolio_market_dict)
            efinance_portfolio_daily_df[['board_type', 'price_limit_rate']] = 'ETF', 0.1
            efinance_portfolio_daily_df.loc[efinance_portfolio_daily_df['股票名称'].str.contains('科创|创业|双创'), 'price_limit_rate'] = 0.2
            
        clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_daily_df)
        clean_stock_bj_df = self.clean_efinance_get_quote_history(efinance_stock_bj_daily_df)
        clean_portfolio_df = self.clean_efinance_get_quote_history(efinance_portfolio_daily_df)
        
        # 数据来源表
        clean_stock_sh_sz_df['source_table'] = 'ods_ohlc_incr_baostock_stock_sh_sz_daily'
        clean_stock_bj_df['source_table'] = 'ods_ohlc_incr_efinance_stock_bj_daily'
        clean_portfolio_df['source_table'] = 'ods_ohlc_incr_efinance_portfolio_daily'
        
        print('2', clean_stock_sh_sz_df.shape)
        print('2bj', clean_stock_bj_df.shape)
        print('2etf', clean_portfolio_df.shape)
        asset_daily_df = pd.concat([clean_stock_sh_sz_df, clean_stock_bj_df, clean_portfolio_df], axis=0)
        print('3', asset_daily_df.shape)
        #return asset_daily_df, clean_stock_sh_sz_df, clean_stock_bj_df, clean_portfolio_df
        #asset_daily_df['avg_price'] = asset_daily_df['turnover'] / asset_daily_df['volume']#每单位成交量的平均价格
        #asset_daily_df['avg_price'] = asset_daily_df['avg_price'].round(4)
        
        # 使用 ThreadPoolExecutor 来并行处理每个股票的数据
        grouped = asset_daily_df.groupby('full_code')
        asset_daily_df = utils_thread.thread(grouped, calculate_prev_close, max_workers=8)
        print('4', asset_daily_df.shape)
        grouped = asset_daily_df.groupby('full_code') # 是两个grouped，否则上一个计算的结果不会输出
        asset_daily_df = utils_thread.thread(grouped, finance_limit.limit_prices, max_workers=8)
        print('5', asset_daily_df.shape)
        ## date
        trading_day_prev_dict = self.trading_day_alignment.prev_trading_day_dict(date_num=1)
        asset_daily_df['prev_date'] = asset_daily_df.date.map(trading_day_prev_dict)
        print('6', asset_daily_df.shape)
        trading_day_next_dict = self.trading_day_alignment.prev_trading_day_dict(date_num=-1)
        asset_daily_df['next_date'] = asset_daily_df.date.map(trading_day_next_dict)
        print('7', asset_daily_df.shape)
        asset_daily_df['volume'] = asset_daily_df['volume'].astype('int64')
        print('8', asset_daily_df.shape)
        # 特征: 日期_距离上一次开盘天数
        asset_daily_df['date_diff_prev'] = (pd.to_datetime(asset_daily_df.prev_date) - pd.to_datetime(asset_daily_df.date)).dt.days
        asset_daily_df['date_diff_prev'] =  asset_daily_df['date_diff_prev'].fillna(-1)
        asset_daily_df['date_diff_next'] = (pd.to_datetime(asset_daily_df.next_date) - pd.to_datetime(asset_daily_df.date)).dt.days
        asset_daily_df['date_diff_next'] =  asset_daily_df['date_diff_next'].fillna(1)
        print('9', asset_daily_df.shape)
        # 特征：日期_星期
        asset_daily_df['date_week'] = pd.to_datetime(asset_daily_df['date'], format='%Y-%m-%d').dt.day_name()
        print('10', asset_daily_df.shape)
        # asset_daily_df = asset_daily_df.groupby('full_code').apply(prev_close)
        # asset_daily_df = asset_daily_df.groupby('full_code').apply(finance_limit.limit_prices)
        # asset_daily_df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # 比较合理的拼接是：国家+股票市场+限价百分比, 用板块+限价百分比比较快速,后面可能加个股票性质是高股息还是妖股、蓝筹股
        asset_daily_df['board_primary_key'] = (asset_daily_df['board_type'].astype(str) +
                                                     asset_daily_df['price_limit_rate'].astype(str)
                                                     # asset_daily_df['is_limit_up_prev'].astype(str) +
                                                     # asset_daily_df['is_limit_down_prev'].astype(str) +
                                                     # asset_daily_df['is_flat_price_prev'].astype(str)
                                                     ).apply(utils_character.md5_str)
        print('11', asset_daily_df.shape)
        float_columns=['date_diff_prev','date_diff_next','price_limit_rate','prev_close','volume','turnover','amplitude',
                       'turnover_pct','chg_rel','price_chg','pe_ttm','ps_ttm','pcf_ttm','pb_mrq',
                       'limit_up','limit_down']#,'avg_price'
        asset_daily_df[float_columns] = asset_daily_df[float_columns].fillna(0)
        print('12', asset_daily_df.shape)
        utils_data.output_database(df,
                                   filename='dwd_ohlc_incr_stock_daily',
                                   if_exists='append',
                                   )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_type', type=str, default='full', help='Data update method')
    args = parser.parse_args()
    
    dwd_stock = DwdStock(freq='d', adj_type='pre')
    if args.update_type=='full':
        asset_daily_df = dwd_stock.stock_daily_full()
        utils_data.output_database_large(asset_daily_df,
                                         filename='dwd_ohlc_incr_stock_daily2',
                                         if_exists='replace',
                                         )
    elif args.update_type=='incr':
        dwd_stock.stock_daily_incr()
        
# =============================================================================
#     a_stock_k_raw_df.columns
#     Out[2]: 
#     Index(['date', 'asset_code', 'open', 'high', 'low', 'close', 'prev_close', 'volume',
#            'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
#            'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'insert_timestamp',
#            'primary_key'],
#           dtype='object')
# 
#     a_stock_k_df.columns
#     Out[3]: 
#     Index(['primary_key', 'date', 'asset_code', 'code_name', 'open', 'high', 'low','close', 'volume','amount','turn','pct_chg', 
#             'prev_close',  'adjustflag', 
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

