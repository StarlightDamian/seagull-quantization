# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 01:43:15 2024

@author: awei
短期推荐测试(test_3_short_term_recommend)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database
from test_ import test_0_lightgbm, test_2_stock_pick

TASK_NAME = 'short_term_recommend'
TEST_TABLE_NAME = 'test_3_short_term_recommend'
#MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_regression_short_term_recommend.joblib'
TARGET_PRED_NAMES = ['rear_next_rise_pct_pred', 'rear_next_fall_pct_pred','rear_next_pct_pred']


class testShortTermRecommend(test_0_lightgbm.lightgbmTest):
    def __init__(self, multioutput_model_path=None):
        super().__init__(TARGET_PRED_NAMES)
        self.test_table_name = TEST_TABLE_NAME
        self.target_pred_names = TARGET_PRED_NAMES
        self.multioutput_model_path = multioutput_model_path
        self.task_name = TASK_NAME
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for testing')
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='Start time for testing')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for testing')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)

    print(history_day_df)
    
    test_stock_pick = test_2_stock_pick.testStockPick()
    history_day_board_df = test_stock_pick.board_data(history_day_df)
    prediction_stock_pick_df = test_stock_pick.test_board_pipline(history_day_board_df)
    
    history_day_board_df = pd.merge(history_day_board_df, prediction_stock_pick_df[['primary_key', 'rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'rear_open_pct_pred', 'rear_close_pct_pred']])
    test_short_term_recommend = testShortTermRecommend()
    prediction_df = test_short_term_recommend.test_board_pipline(history_day_board_df)
    
    #prediction_df.to_csv(f'{path}/data/backtest_2_stock_pick.csv',index=False)
    #eval_stock_pick(prediction_df)