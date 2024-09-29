# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
选股测试(test_2_stock_pick)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database, base_trading_day, base_utils
from test_ import test_0_lightgbm

TASK_NAME = 'stock_pick'
TEST_TABLE_NAME = 'test_2_stock_pick'
#MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_regression_stock_pick.joblib'
TARGET_PRED_NAMES = ['rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'rear_open_pct_pred', 'rear_close_pct_pred']


class testStockPick(test_0_lightgbm.lightgbmTest):
    def __init__(self, multioutput_model_path=None):
        super().__init__(TARGET_PRED_NAMES)
        self.test_table_name = TEST_TABLE_NAME
        self.target_pred_names = TARGET_PRED_NAMES
        self.multioutput_model_path = multioutput_model_path
        self.task_name = TASK_NAME
        
def eval_stock_pick(prediction_df):
    trading_day = base_trading_day.tradingDay()
    trading_day_after_dict = trading_day.specified_trading_day_after()
    prediction_df['date_after'] = prediction_df['date'].map(trading_day_after_dict)
    
    rear_day_df = prediction_df[['date_after','code','code_name','open', 'high', 'low',
    'close']]
    rear_day_df = rear_day_df.rename(columns={'open': 'rear_open_real',
                                            'low': 'rear_low_real',
                                            'high': 'rear_high_real',
                                            'close': 'rear_close_real',
                                            })
    
    rear_day_df['primary_key'] = (rear_day_df['date_after']+rear_day_df['code']).apply(base_utils.md5_str)
    
    prediction_df['rear_open_pred'] = ((prediction_df['rear_open_pct_pred'] / 100)+1)*prediction_df['close']

    prediction_df['rear_low_pred'] = ((prediction_df['rear_low_pct_pred'] / 100)+1)*prediction_df['close']

    prediction_df['rear_high_pred'] = ((prediction_df['rear_high_pct_pred'] / 100)+1)*prediction_df['close']

    prediction_df['rear_close_pred'] = ((prediction_df['rear_close_pct_pred'] / 100)+1)*prediction_df['close']    
    
    prediction_df = prediction_df[['date','rear_open_pred', 'rear_low_pred', 'rear_high_pred', 'rear_close_pred', 'primary_key']]
    
    
    eval_df = pd.merge(prediction_df, rear_day_df, on='primary_key')
    eval_df = eval_df[['rear_open_real','rear_open_pred','rear_low_real', 'rear_low_pred','rear_high_real', 'rear_high_pred','rear_close_real', 'rear_close_pred','code','code_name','date']]
    eval_df.to_csv(f'{path}/data/backtest_2_stock_pick_eval.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2024-03-01', help='Start time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2024-03-06', help='End time for backtesting')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)

    print(history_day_df)
    
    test_stock_pick = testStockPick()
    history_day_df = test_stock_pick.board_data(history_day_df) # board_type, price_limit_pct
    #prediction_df = test_stock_pick.test_pipline(history_day_df)
    _ = test_stock_pick.test_board_pipline(history_day_df)#prediction_df
    #prediction_df.to_csv(f'{path}/data/backtest_2_stock_pick.csv',index=False)
    
    #eval_stock_pick(prediction_df)
    
