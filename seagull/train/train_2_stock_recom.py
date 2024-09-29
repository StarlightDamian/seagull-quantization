# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 01:27:29 2024

@author: awei
推票训练(train_2_stock_recommended)
"""
import argparse
import pandas as pd

from __init__ import path
from train import train_1_lightgbm_regression
from feature_engineering import feature_engineering_main
from base import base_connect_database

TRAIN_TABLE_NAME = 'train_2_stock_recommend'
TARGET_REAL_NAMES = ['rear_rise_pct_real']  
PREDICTION_CSV_PATH = f'{path}/data/train_stock_recommend.csv'
MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/lightgbm_regression_stock_recommend.joblib'
#EVAL_TABLE_NAME = 'eval_train'


class trainStockRecommend(train_1_lightgbm_regression.lightgbmRegressionTrain):
    def __init__(self):
        """
        Initialize standardStockPickTrain object, including feature engineering and database connection.
        """
        super().__init__()
        self.feature_engineering = feature_engineering_main.featureEngineering(TARGET_REAL_NAMES)
        self.train_table_name = TRAIN_TABLE_NAME
        self.target_pred_names = TARGET_REAL_NAMES
        self.multioutput_model_path = MULTIOUTPUT_MODEL_PATH
        
        #train_model
        self.task_name = 'stock_recommend'
        
        # 预测的特征是否可以作为推票的特征进行训练
        #rear_close_pct_real	rear_diff_pct_real	rear_high_pct_real	rear_low_pct_real	rear_open_pct_real	rear_price_limit
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2016-03-01', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2024-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2024-02-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # Load date range data
    #history_day_df = data_loading.feather_file_merge(args.date_start, args.date_end)
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    recommend_df = pd.read_csv(f'{path}/data/rear_rise_pct_real_df_20160301_20240201.csv')
    recommend_df = recommend_df[~(recommend_df.rear_rise_pct_real.isnull())]
    recommend_df = recommend_df.rename(columns={'primary_key_date': 
                                                    'primary_key'})
    recommend_merge_df = pd.merge(recommend_df[['primary_key', 'rear_rise_pct_real']], history_day_df, on='primary_key')
    
    train_stock_recommend = trainStockRecommend()#date_start=args.date_start, date_end=args.date_end
    prediction_df = train_stock_recommend.train_pipline(recommend_merge_df)
    
    # Rename and save to CSV file
    prediction_df = prediction_df.rename(
        columns={'rear_rise_pct_real': '明天_上升价幅_真实值',
                 #'rear_rise_pct_pred': '明天_上升价幅_预测值',
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
                 'primary_key': '信息主键编号',
                 'date': '日期',
                 'code': '股票代码',
                 'code_name': '股票中文名称',
                 'preclose': '昨日收盘价',
                 'isST': '是否ST',
                 'insert_timestamp': '信息入库时间',
                })
    prediction_df.to_csv(PREDICTION_CSV_PATH, index=False)
    
    #stock_model.load_model(MODEL_PATH)
    
    # Plot feature importance
    #stock_model.plot_feature_importance()
    #stock_model.plot_feature_importance()


