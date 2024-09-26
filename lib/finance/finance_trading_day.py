# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:52:15 2024

@author: awei
交易日(base_trading_day)
"""
import numpy as np
import pandas as pd

from __init__ import path
from base import base_connect_database


class tradingDay:
    def __init__(self):
        with base_connect_database.engine_conn('postgre') as conn:
            trading_day_df = pd.read_sql('trade_dates', con=conn.engine)
        self.trading_day_df = trading_day_df[trading_day_df.is_trading_day=='1']
        
    def specified_trading_day_before(self, pre_date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param pre_date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        如：'2019-02-12': '2019-02-11'
        """
        trading_day_df = self.trading_day_df
        trading_day_df['rear_date'] = np.insert(trading_day_df.calendar_date, 0, ['']*pre_date_num)[:-pre_date_num]
        trading_day_pre_dict = trading_day_df.set_index('calendar_date')['rear_date'].to_dict()
        return trading_day_pre_dict

    def specified_trading_day_after(self, pre_date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param pre_date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        如：'2019-02-11': '2019-02-12'
        """
        trading_day_df = self.trading_day_df
        trading_day_df['pre_date'] = np.insert(trading_day_df.calendar_date, trading_day_df.shape[0], ['']*pre_date_num)[pre_date_num:]
        
        trading_day_pre_dict = trading_day_df.set_index('calendar_date')['pre_date'].to_dict()
        return trading_day_pre_dict