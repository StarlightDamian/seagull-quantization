# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 01:43:15 2024

@author: awei
选股评估(valid_3_stock_pick)
"""
import argparse

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from seagull.settings import PATH
from train import train_2_stock_pick
from base import base_connect_database
from valid import valid_0_lightgbm
from test_ import test_2_stock_pick

TASK_NAME = 'price_high'
VALID_TABLE_NAME = 'valid_2_stock_pick'
#MULTIOUTPUT_MODEL_PATH = f'{PATH}/checkpoint/lightgbm_regression_stock_pick.joblib'
TARGET_NAMES = ['rear_low_rate', 'rear_high_rate']

class validStockPick(valid_0_lightgbm.lightgbmValid):
    def __init__(self):
        super().__init__(TARGET_NAMES)
        self.valid_table_name = VALID_TABLE_NAME
        self.target_names = TARGET_NAMES
        self.multioutput_model_path = None  # MULTIOUTPUT_MODEL_PATH
        self.task_name = TASK_NAME
        
        self.train_stock_pick = train_2_stock_pick.trainStockPick()
        self.test_stock_pick = test_2_stock_pick.testStockPick()
        
    def grid_search(self, history_day_df):
        # 定义参数空间
        param_dist = {
            'estimator__num_leaves': randint(6, 200),  # 这里 'estimator__' 表示在包装器中的参数
            'estimator__learning_rate': [0.01,0.02,0.03,0.05,0.08, 0.1, 0.13,0.16,0.2,0.23],
            'estimator__metric': ['mae','root_mean_squared_error'],
            'estimator__verbose': [-1],
            'estimator__n_estimators': randint(100, 1000),
            'estimator__early_stop_round': randint(20, 100),
        }
        
        x_train, x_test, y_train, y_test = self.feature_engineering_split(history_day_df)
    
        # 使用 RandomizedSearchCV 进行随机搜索
        random_search = RandomizedSearchCV(self.model_multioutput, param_distributions=param_dist, n_iter=50, cv=5, scoring='neg_mean_absolute_error')
        
        del x_train['primary_key']
        random_search.fit(x_train, y_train)
          
        # 输出最佳参数
        print("Best parameters:", random_search.best_params_)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-12-01', help='Start time for testing')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for testing')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn("POSTGRES") as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)

    print(history_day_df)
    
    #test_stock_pick = test_2_stock_pick.testStockPick()
    #prediction_df = test_stock_pick.test_board_pipline(history_day_df)
    #history_day_df = pd.merge(history_day_df, prediction_df[['primary_key', ]])
    
    
    valid_stock_pick = validStockPick()
    valid_stock_pick.valid_board_pipeline(history_day_df)
    
    #valid_stock_pick.grid_search(history_day_df)
    
    #prediction_df = valid_stock_pick.test_pipline(history_day_df)
    #prediction_df.to_csv(f'{PATH}/data/prediction_df.csv',index=False)



