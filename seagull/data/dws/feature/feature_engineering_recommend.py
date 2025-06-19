# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:13:43 2024

@author: awei
推票数据预处理(feature_engineering_recommended)
"""
import argparse

import numpy as np
import pandas as pd

from seagull.settings import PATH
from base import base_connect_database, base_utils
from get_data import data_history_a_stock_5_min
from datetime import datetime, timedelta
class stockRecommend:
    def __init__(self):
        with base_connect_database.engine_conn("POSTGRES") as conn:
            trading_day_df = pd.read_sql('trade_dates', con=conn.engine)
        self.trading_day_df = trading_day_df[trading_day_df.is_trading_day=='1']
        
    def specified_trading_day(self, pre_date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param pre_date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        """
        trading_day_df = self.trading_day_df
        trading_day_df['rear_date'] = np.insert(trading_day_df.calendar_date, 0, ['']*pre_date_num)[:-pre_date_num]
        trading_day_pre_dict = trading_day_df.set_index('calendar_date')['rear_date'].to_dict()
        return trading_day_pre_dict
    
    def pipline(self, history_day_df):
        
        print(4,datetime.now().strftime("%F %T"))
        trading_day_target_dict = self.specified_trading_day(pre_date_num=1)
        history_day_df['target_date'] = history_day_df.date.map(trading_day_target_dict)
        history_day_df['primary_key_date'] = (history_day_df['target_date']+history_day_df['code']).apply(base_utils.md5_str)
        print(5,datetime.now().strftime("%F %T"))
        history_day_df['rear_rise_pct_real'] = np.nan
        
        day_df = history_day_df.groupby('primary_key_date').agg(
            max_high=('high', 'max'),
            min_low=('low', 'min'),
            idxmax = ('high', 'idxmax'),
            idxmin = ('low', 'idxmin'),
            ).reset_index()
        print(6,datetime.now().strftime("%F %T"))
        day_df['idxmax_time'] = history_day_df.loc[day_df['idxmax'], 'time'].values
        day_df['idxmin_time'] = history_day_df.loc[day_df['idxmin'], 'time'].values
        print(7,datetime.now().strftime("%F %T"))
        day_df['rear_rise_pct_real'] = np.where(day_df['idxmax_time'] > day_df['idxmin_time'], 
                                                ((day_df['max_high'] - day_df['min_low']) / day_df['min_low']) * 100,
                                                ((day_df['min_low'] - day_df['max_high']) / day_df['min_low']) * 100)
        print(8,datetime.now().strftime("%F %T"))
        day_df.loc[(day_df['min_low']==day_df['max_high']) | (day_df['idxmax_time']==day_df['idxmin_time']), 'rear_rise_pct_real'] = np.nan

        print(9,datetime.now().strftime("%F %T"))
        day_df = day_df[~(day_df.rear_rise_pct_real.isnull())]
        data_info_df = history_day_df[['date', 'code', 'target_date', 'primary_key_date']].drop_duplicates('primary_key_date',keep='first')
        print(10,datetime.now().strftime("%F %T"))
        day_df = pd.merge(day_df, data_info_df, on='primary_key_date').reset_index(drop=True)
        day_df = day_df[['date', 'code', 'target_date', 'primary_key_date', 'rear_rise_pct_real']]
        print(11,datetime.now().strftime("%F %T"))
        return day_df
        #return rear_rise_pct_real_df[['date', 'code', 'target_date', 'primary_key_date',
        #'rear_rise_pct_real']].reset_index(drop=True).drop_duplicates('primary_key_date',keep='first')
    
if __name__ == '__main__':
    #with base_connect_database.engine_conn("POSTGRES") as conn:
    #    history_day_df = pd.read_sql("SELECT * FROM history_a_stock_5_min", con=conn.engine)# WHERE date = '{now}'
    #history_day_df = pd.read_feather(f'{PATH}/_file/history_a_stock_5_min.feather').reset_index(drop=True)
    
    #history_day_df = history_day_df[history_day_df.code.isin(['sh.600826', 'sh.600827', 'sh.600828'])]
    
    date = '2016-03-01_2017-01-01'#'2016-03-01_2017-01-01'#'2022-01-01_2022-10-10'
    get_day_data = data_history_a_stock_5_min.Get5MinData()
    history_day_df = get_day_data.add_new_data(f'{PATH}/_file/history_a_stock_5_min/{date}/')
    
    history_day_df[['low', 'high']] = history_day_df[['low', 'high']].astype(float)
    stock_recommend = stockRecommend()
    day_df = stock_recommend.pipline(history_day_df)
    day_df.to_csv(f'{PATH}/_file/history_a_stock_5_min/rise_df_{date}.csv', index=False)
    