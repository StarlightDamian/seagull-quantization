# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:53:45 2024

@author: awei
训练涨跌停分类模型(train_2_price_limit)
"""
import argparse
import pandas as pd

from __init__ import path
from train import train_1_lightgbm_classification
from feature_engineering import feature_engineering_main
from base import base_connect_database

TRAIN_TABLE_NAME = 'train_2_price_limit'
TARGET_REAL_NAMES = ['price_limit']

class trainPriceLimit(train_1_lightgbm_classification.lightgbmClassificationTrain):
    def __init__(self):
        super().__init__()
        self.feature_engineering = feature_engineering_main.featureEngineering(TARGET_REAL_NAMES)
        self.train_table_name = TRAIN_TABLE_NAME
        self.target_pred_names = TARGET_REAL_NAMES
        
        #train_model
        self.task_name = 'price_limit'
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2018-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2020-01-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
        
    train_price_limit = trainPriceLimit()
    prediction_df = train_price_limit.train_pipline(history_day_df)