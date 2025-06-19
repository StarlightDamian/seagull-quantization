# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 04:48:12 2024

@author: awei
评估(application_eval)
"""
import argparse

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from seagull.settings import PATH
from base import base_connect_database, base_trading_day
from application import application_rl_real

scaler = MinMaxScaler(feature_range=(0, 10))
pd.set_option('display.max_columns', 15)  # 显示 15 列
RE_ENVIRONMENT = 'rl_environment'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_date', type=str, default='2024-03-27', help='Start time for training')
    args = parser.parse_args()
    
    trading_day = base_trading_day.tradingDay()
    trading_day_after_dict = trading_day.specified_trading_day_after()
    trading_day_next = trading_day_after_dict[args.target_date]
    
    prediction_merge_df = application_rl_real.prediction_target_date(args.target_date)
    prediction_df = prediction_merge_df[['code', 'code_name','industry','close','vote_index','rear_open_pred', 'rear_low_pred', 'rear_high_pred', 'rear_close_pred']]#, 'rear_close_pct_pred','rear_next_pct_pred'
    
    #print(trading_day_next)
    with base_connect_database.engine_conn("POSTGRES") as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date = '{trading_day_next}'", con=conn.engine)
        if history_day_df.empty:
            print('历史数据为空')
    history_day_df = history_day_df[['code','open', 'low', 'high','close']]
    history_day_df = history_day_df.rename(columns={'open': 'rear_open',
                                                    'low': 'rear_low',
                                                    'high': 'rear_high',
                                                    'close': 'rear_close',
                                                    })
    
    prediction_df = pd.merge(prediction_df, history_day_df, on='code')
    prediction_df['rear_close_pct_real'] = ((prediction_df['rear_close'] - prediction_df['close']) / prediction_df['close']) * 100
    
    prediction_df[['rear_close_pct_real', 'vote_index']] = prediction_df[['rear_close_pct_real', 'vote_index']].round(2)
    
    target_date_replace = args.target_date.replace('-', '')
    prediction_df = prediction_df[['code', 'code_name', 'industry', 'close','rear_open', 'rear_open_pred', 'rear_low','rear_low_pred', 'rear_high','rear_high_pred', 'rear_close','rear_close_pred',
           'rear_close_pct_real', 'vote_index']]
    prediction_rename_df = prediction_df.rename(columns={
        'code': '股票代码',
        'code_name': '公司名称',
        'industry': '行业',
        'close': '前交易日_收盘价',
        'vote_index': '推荐指数',
        'rear_open': '真实_开盘价',
        'rear_open_pred': '预测_开盘价',
        'rear_low': '真实_最低价',
        'rear_low_pred': '预测_最低价',
        'rear_high': '真实_最高价',
        'rear_high_pred': '预测_最高价',
        'rear_close': '真实_收盘价',
        'rear_close_pred': '预测_收盘价',
        'rear_close_pct_real': '涨跌幅',
        })
    prediction_rename_df.to_csv(f'{PATH}/wechat/复盘_{target_date_replace}.csv', index=False)