# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:01:48 2024

@author: awei
训练短期推荐模型(train_3_short_term_recommend_board)
"""
import argparse
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from __init__ import path
from train import train_1_lightgbm_regression, train_3_short_term_recommend
from base import base_connect_database, base_trading_day, base_utils

#EVAL_TABLE_NAME = 'eval_train'
TRAIN_TABLE_NAME = 'train_3_short_term_recommend'
TARGET_REAL_NAMES = ['rear_next_rise_pct_real', 'rear_next_fall_pct_real', 'rear_next_pct_real']
#MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_regression_short_term_recommend.joblib'
PREDICTION_PRICE_OUTPUT_CSV_PATH = f'{path}/data/train_short_term_recommend.csv'


class trainShortTermRecommendBoard(train_1_lightgbm_regression.lightgbmRegressionTrain):
    def __init__(self):
        super().__init__(TARGET_REAL_NAMES)
        self.train_table_name = TRAIN_TABLE_NAME
        self.target_pred_names = TARGET_REAL_NAMES
        #self.multioutput_model_path = MULTIOUTPUT_MODEL_PATH
        
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 127, #37,96 决策树上的叶子节点的数量，控制树的复杂度
            'learning_rate': 0.08,  # 0.05,0.1
            'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
            #w×RMSE+(1−w)×MAE
            'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
            'max_depth': 7,
            'n_estimators': 1000,
            #'early_stopping_round':50,
            'min_child_sample':40,
            'min_child_weight':1,
            'subsample':0.8,
            'colsample_bytree':0.8,
        }
        # loading data
        lgb_regressor = lgb.LGBMRegressor(**params)
        self.model_multioutput = MultiOutputRegressor(lgb_regressor)
    
        #train_model
        self.task_name = 'short_term_recommend'
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for training')
    parser.add_argument('--date_start', type=str, default='2022-06-01', help='Start time of training')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='end time of training')
    #parser.add_argument('--date_end', type=str, default='2024-03-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # 根据数据库该表test_stock_pick的日期进行二次训练
    history_day_short_term_df = train_3_short_term_recommend.feature_engineering_short_term_recommend(args.date_start, args.date_end)
    
    train_stock_term_recommend_board = trainShortTermRecommendBoard()
    #train_stock_term_recommend_board.grid_search(history_day_short_term_df)
    history_day_short_term_df = train_stock_term_recommend_board.board_data(history_day_short_term_df)
    prediction_df = train_stock_term_recommend_board.train_board_pipline(history_day_short_term_df)
    
    # Rename and save to CSV file
    prediction_df = prediction_df.rename(
        columns={'rear_next_rise_pct_real': '明天_间隔日上升价幅_真实值',
                 'rear_next_fall_pct_real': '明天_间隔日下降价幅_真实值',
                 'rear_open_pct_pred': '明天_开盘价幅_预测值',
                 'rear_low_pct_pred': '明天_最低价幅_预测值',
                 'rear_high_pct_pred': '明天_最高价幅_预测值',
                 'rear_close_pct_pred': '明天_收盘价幅_预测值',
                 'rear_diff_pct_pred': '明天_变化价幅_预测值',
                 'open': '今开盘价格',
                 'high': '最高价',
                 'low': '最低价',
                 'close': '今收盘价',
                 'volume': '成交数量',
                 'amount': '成交金额',
                 'turn': '换手率',
                 'macro_amount': '深沪成交额',
                 'macro_amount_diff_1': '深沪成交额增量',
                 'pctChg': '涨跌幅',
                 'remarks': '备注',
                 'date': '日期',
                 'code': '股票代码',
                 'code_name': '股票中文名称',
                 'isST': '是否ST',
                })
    prediction_df.to_csv(PREDICTION_PRICE_OUTPUT_CSV_PATH, index=False)
