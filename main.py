# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:28:26 2023

@author: awei
"""
from datetime import datetime

import pandas as pd

from __init__ import path
from base import base_connect_database
from application import application_daily, application_rl_real
from tests import test_board

class dailyPipline:
    def __init__(self):
        ...
        
    def pipline_get_data():
        application_daily.custom_date()

    def pipline_feature_engineering_fredict(date_start, date_end='2029-01-01'):
        with base_connect_database.engine_conn('postgre') as conn:
            history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{date_start}' AND date < '{date_end}'", con=conn.engine)
        
        print(history_day_df)
        board = test_board.testBoard()
        history_day_df = board.board_data(history_day_df)
        prediction_df = board.board_pipline(history_day_df)
        prediction_df.to_sql('rl_environment', con=conn.engine, index=False, if_exists='append')
        
    def pipline_stock_pick():
        ...
    
    def wechat():
        ...
        
if __name__ == '__main__':
    today = datetime.now().strftime('%F')
    
    daily_pipline = dailyPipline()
    daily_pipline.pipline_get_data()
    daily_pipline.pipline_feature_engineering_fredict(today)
    #daily_pipline.pipline_stock_pick_wechat(today)