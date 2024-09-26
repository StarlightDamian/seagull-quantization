# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
推荐测试(test_recommend)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database
from test_ import test_0_lightgbm
TEST_TABLE_NAME = 'test_1_stock_recommend'
MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_regression_stock_recommend.joblib'
TARGET_PRED_NAMES = ['rear_rise_pct_pred']


class testStockRecommend(test_0_lightgbm.lightgbmTest):
    def __init__(self):
        super().__init__(TARGET_PRED_NAMES)
        self.test_table_name = TEST_TABLE_NAME
        self.target_pred_names = TARGET_PRED_NAMES
        self.multioutput_model_path = MULTIOUTPUT_MODEL_PATH
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    parser.add_argument('--date_start', type=str, default='2024-02-08', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2024-03-06', help='End time for backtesting')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(history_day_df)
    
    test_stock_recommend = testStockRecommend()
    prediction_df = test_stock_recommend.test_pipline(history_day_df)
    
    #prediction_df.rear_rise_pct_pred.mean()
    #prediction_df[prediction_df.date=='2024-03-01']
    #prediction_df.groupby('date').agg(maxid=('rear_rise_pct_pred','max'))