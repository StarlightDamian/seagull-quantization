# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:43:31 2024

@author: awei
交易日的额外操作(finance_trading_day)
"""
import pandas as pd

from __init__ import path
from utils import utils_database, utils_data
from data import ods_info_incr_efinance_trading_day

class TradingDayAlignment(ods_info_incr_efinance_trading_day.TradingDay):
    def __init__(self):
        if not utils_data.table_in_database('dwd_info_incr_trading_day'):
            self.pipeline()
        with utils_database.engine_conn('postgre') as conn:
            trading_day_df = pd.read_sql('dwd_info_incr_trading_day', con=conn.engine)
        self.trading_day_df = trading_day_df[(trading_day_df.trade_status=='1')|(trading_day_df.trade_status==1)]
        
    def prev_trading_day_dict(self, date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        如：'2019-02-12': '2019-02-11'
        """
        trading_day_df = self.trading_day_df
        # print(trading_day_df.columns)
        trading_day_df['next_date'] = trading_day_df['date'].shift(date_num)
        return dict(zip(trading_day_df['date'], trading_day_df['next_date']))

    def next_trading_day_dict(self, date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        如：'2019-02-11': '2019-02-12'
        """
        trading_day_df = self.trading_day_df
        trading_day_df['prev_date'] = trading_day_df['date'].shift(-date_num)
        return dict(zip(trading_day_df['date'], trading_day_df['prev_date']))
    
    def shift_day(self, date_start, date_num=1):
        """
        用于刷特征用，获取bar的前N个日期真实日期
        :param date_num: 前置日期数，默认为1
        :return: 前置日期
        输入示例: date_start='2019-02-12'
        输出示例：'2019-02-11'
        """
        trading_day_df = self.trading_day_df
        trading_day_df['next_date'] = trading_day_df['date'].shift(date_num)
        return trading_day_df.loc[trading_day_df.date<=date_start, 'next_date'].tail(1).values[0]
        
    
if __name__ == '__main__':
    trading_day_alignment = TradingDayAlignment()
    date_start_prev = trading_day_alignment.shift_day(date_start='2025-01-05', date_num=5)
    
    