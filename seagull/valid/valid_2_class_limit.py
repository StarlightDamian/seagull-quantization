# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 01:43:15 2024

@author: awei
短期推荐评估(valid_3_short_term_recommend)
"""
import argparse

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from seagull.settings import PATH
from base import base_connect_database
from test_ import test_2_stock_pick, test_2_short_term_recommend
from valid import valid_0_lightgbm
from train import train_3_short_term_recommend

TASK_NAME = 'short_term_recommend'
VALID_TABLE_NAME = 'valid_3_short_term_recommend'
#MULTIOUTPUT_MODEL_PATH = f'{PATH}/checkpoint/lightgbm_regression_short_term_recommend.joblib'
TARGET_PRED_NAMES = ['rear_next_pct_pred']
TARGET_REAL_NAMES = ['rear_next_pct_real']

class validShortTermRecommend(valid_0_lightgbm.lightgbmValid):
    def __init__(self):
        super().__init__(TARGET_PRED_NAMES)
        #self.feature_engineering = eval_0_lightgbm.featureEngineeringEval(TARGET_PRED_NAMES)
        #self.feature_engineering = feature_engineering_main.featureEngineering(TARGET_PRED_NAMES)
        self.valid_table_name = VALID_TABLE_NAME
        self.target_pred_names = TARGET_PRED_NAMES
        self.target_real_names = TARGET_REAL_NAMES
        self.multioutput_model_path = None #multioutput_model_path #MULTIOUTPUT_MODEL_PATH
        self.task_name = TASK_NAME
    
    def grid_search(self, history_day_df):
        # 定义参数空间
        param_dist = {
            'estimator__num_leaves': randint(6, 100),  # 这里 'estimator__' 表示在包装器中的参数
            'estimator__learning_rate': [0.01,0.03,0.06, 0.1, 0.2],
            'estimator__metric': ['mae','root_mean_squared_error'],
            'estimator__verbose': [-1],
        }
        
        x_train, x_test, y_train, y_test = self.feature_engineering_split(history_day_df)
    
        # 使用 RandomizedSearchCV 进行随机搜索
        random_search = RandomizedSearchCV(self.model_multioutput, param_distributions=param_dist, n_iter=30, cv=4, scoring='neg_mean_absolute_error')
        
        del x_train['primary_key']
        random_search.fit(x_train, y_train)
        
        # 输出最佳参数
        print("Best parameters:", random_search.best_params_)
        
    #def __apply_eval_board(self, suntable):
    #    if not suntable.empty:
            #board_type = suntable.name
            #print('board_type', board_type)
            #global suntable1
            #suntable1 = suntable
            #_ = self.train_pipline(suntable, board_type=board_type)
    

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='Start time for testing')
    #parser.add_argument('--date_start', type=str, default='2021-12-01', help='Start time for testing')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for testing')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    #with base_connect_database.engine_conn("POSTGRES") as conn:
    #    history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    history_day_df = train_3_short_term_recommend.feature_engineering_short_term_recommend(args.date_start, args.date_end)



    
    
    print(history_day_df)
    
    valid_short_term_recommend = validShortTermRecommend()
    valid_short_term_recommend.valid_board_pipline(history_day_df)
    
    #eval_short_term_recommend.grid_search(history_day_df)
    
# =============================================================================
#     test_stock_pick = test_2_stock_pick.testStockPick()
#     prediction_stock_pick_df = test_stock_pick.test_pipline(history_day_df)
#     
#     history_day_df = pd.merge(history_day_df, prediction_stock_pick_df[['primary_key', 'rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'rear_open_pct_pred', 'rear_close_pct_pred']])
#     test_short_term_recommend = test_2_short_term_recommend.testShortTermRecommend()
#     prediction_df = test_short_term_recommend.test_pipline(history_day_df)
#     print(prediction_df.columns)
#     
#     train_df = train_2_short_term_recommend.feature_engineering_short_term_recommend(args.date_start, args.date_end)
#     
#     prediction_handle_df = prediction_df[['primary_key'] + TARGET_PRED_NAMES]
#     train_handle_df = train_df[['primary_key', 'date', 'code', 'code_name']+TARGET_REAL_NAMES]
#     handle_df = pd.merge(train_handle_df, prediction_handle_df, on='primary_key')
#     handle_df = handle_df.drop_duplicates('primary_key',keep='first')
#     
#     eval_short_term_recommend = evalShortTermRecommend()
#     eval_df, eval_details_df = eval_short_term_recommend.eval_pipline(handle_df)
#     
#     eval_details_df.to_csv(f'{PATH}/data/eval_details_df.csv',index=False)
#     #eval_df.to_csv(f'{PATH}/data/backtest_2_stock_pick.csv',index=False)
#     #eval_stock_pick(eval_df)
# 
# 
# 
# 
# =============================================================================
