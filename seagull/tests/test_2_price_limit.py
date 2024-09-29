# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
推荐测试(test_2_price_limit)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database
from test_ import test_0_lightgbm
TASK_NAME = 'price_limit'
TEST_TABLE_NAME = 'test_2_price_limit'
#MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_classification_price_limit.joblib'
TARGET_PRED_NAMES = ['rear_price_limit_pred']


class testPriceLimit(test_0_lightgbm.lightgbmTest):
    def __init__(self, multioutput_model_path=None):
        super().__init__(TARGET_PRED_NAMES)
        self.test_table_name = TEST_TABLE_NAME
        self.target_pred_names = TARGET_PRED_NAMES
        self.multioutput_model_path = multioutput_model_path#MULTIOUTPUT_MODEL_PATH
        self.task_name = TASK_NAME
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    #parser.add_argument('--date_start', type=str, default='2024-02-08', help='Start time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2024-03-06', help='End time for backtesting')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(history_day_df)
    
    test_price_limit = testPriceLimit()
    history_day_df = test_price_limit.board_data(history_day_df)
    _ = test_price_limit.test_board_pipline(history_day_df)
    
    #prediction_df[prediction_df.date=='2024-03-01']
    #prediction_df.groupby('date').agg(maxid=('rear_rise_pct_pred','max'))