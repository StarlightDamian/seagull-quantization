# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:24:08 2024

@author: awei
N倍的负数激活函数(train_2_neg_x_activation)
"""
import argparse
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from __init__ import path
from train import train_1_lightgbm_regression
from feature_engineering import feature_engineering_main
from base import base_connect_database

TRAIN_TABLE_NAME = 'train_2_stock_pick'
TARGET_REAL_NAMES = ['rear_low_pct_real', 'rear_high_pct_real', 'rear_diff_pct_real', 'rear_open_pct_real', 'rear_close_pct_real']
TRAIN_CSV_PATH = f'{path}/data/train_neg_x_activation.csv'


class trainNegXActivation(train_1_lightgbm_regression.lightgbmRegressionTrain):
    def __init__(self):
        super().__init__()
        self.feature_engineering = feature_engineering_main.featureEngineering(TARGET_REAL_NAMES)
        self.train_table_name = TRAIN_TABLE_NAME
        self.target_pred_names = TARGET_REAL_NAMES
        
        ## train_model
        self.task_name = 'neg_x_activation'
        
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 127,  # 决策树上的叶子节点的数量，控制树的复杂度
            'learning_rate': 0.08,  # 0.05,0.1
            'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
            #w×RMSE+(1−w)×MAE
            'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
            #'early_stop_round':20,
            'max_depth':7,
            'n_estimators': 1000,
            'min_child_sample':40,
            'min_child_weight':1,
            'subsample':0.8,
            'colsample_bytree':0.8,
        }
        # loading data
        lgb_regressor = lgb.LGBMRegressor(**params)
        self.model_multioutput = MultiOutputRegressor(lgb_regressor)

def custom_activation(x, n=2):
    """
    自定义激活函数：
    - 当 x >= 0 时，返回 0
    - 当 x < 0 时，返回 n * x
    """
    return np.where(x >= 0, x, n * x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2014-01-01', help='Start time for training')
    parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2020-01-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # Load date range data
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM dwd_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    history_day_df.drop_duplicates('primary_key',keep='first', inplace=True)
    train_neg_x_activation = trainNegXActivation()
    prediction_df = train_neg_x_activation.train_pipline(history_day_df)
    
    # Rename and save to CSV file
    # prediction_df = prediction_df.rename(

    prediction_df.to_csv(TRAIN_CSV_PATH, index=False)

    x = np.array([5, -3, 0, -1])
    y = custom_activation(x)
    print(y)