# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:53:45 2024

@author: awei
训练基于指数分类模型(train_2_class_base_index)
"""
import os
import argparse
import pandas as pd

from __init__ import path
from train import train_1_lightgbm_classification
from utils import utils_database, utils_log

TARGET_NAMES = ['next_based_index_class']
PATH_CSV = f'{path}/data/train_class_base_index.csv'

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

class TrainBaseIndex(train_1_lightgbm_classification.LightgbmClassificationTrain):
    def __init__(self):
        super().__init__()
        self.target_names = TARGET_NAMES
        
        #train_model
        self.task_name = 'class_base_index'
        
        self.params = {'task': 'train',
                'boosting': 'gbdt',
                'objective': 'multiclass',
                'num_class': 3,
                'max_depth':7,
                'num_leaves': 127,
                'learning_rate': 0.1,
                'metric': ['multi_logloss'],
                'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
                #'early_stop_round':20,
                'n_estimators': 1200,
                #'min_child_sample':40,
                #'min_child_weight':1,
                'subsample':0.9,# 样本采样比例
                'colsample_bytree':0.9, # 每棵树的特征采样比例
                }
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2014-01-01', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2023-04-01', help='Start time for training')
    #parser.add_argument('--date_end', type=str, default='2023-12-01', help='end time of training')
    parser.add_argument('--date_start', type=str, default='2023-01-03', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2024-11-01', help='end time of training')
    args = parser.parse_args()
    
    logger.info(f"""
    task: train_2_class_base_index
    date_start: {args.date_start}
    date_end: {args.date_end}
    """)
    
    # dataset
    with utils_database.engine_conn('postgre') as conn:
        asset_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    asset_df.drop_duplicates('primary_key', keep='first', inplace=True)
    
    train_base_index = TrainBaseIndex()
    valid_raw_df = train_base_index.train_board_pipeline(asset_df, keep_train_model=True)
    
    # output
    valid_df = pd.merge(valid_raw_df, asset_df[['primary_key','next_based_index_class','price_limit_rate','open','high',
                                                'low','close','volume','turnover','turnover_pct','chg_rel','date','full_code','code_name']], how='left', on='primary_key')
    columns_dict = {'open': '开盘价',
                     'high': '最高价',
                     'low': '最低价',
                     'close': '收盘价',
                     'volume': '成交数量',
                     'turnover': '成交金额',
                     'turnover_pct': '换手率',
                     'price_limit_rate': '涨跌停比例',
                     'chg_rel': '涨跌幅',
                     'next_based_index_class': '明天_基于指数分类_真实值',
                     'next_based_index_class_pred': '明天_基于指数分类_预测值',
                     'base_index_-1_prob': '明天_基于指数-1_概率',
                     'base_index_0_prob': '明天_基于指数0_概率',
                     'base_index_1_prob': '明天_基于指数1_概率',
                     'date': '日期',
                     'full_code': '股票代码',
                     'code_name': '公司名称',
                    }
    valid_df['rank_score'] = valid_df['base_index_-1_prob']*0 +\
                        valid_df['base_index_0_prob']*1 +\
                        valid_df['base_index_1_prob']*2
    valid_df = valid_df.sort_values(by='rank_score', ascending=False) 
    
    base_index_columns = ['base_index_-1_prob', 'base_index_0_prob', 'base_index_1_prob']
    valid_df[base_index_columns] = valid_df[base_index_columns]*100
    valid_df[base_index_columns] = valid_df[base_index_columns].round(2)
    
    output_valid_df = valid_df.rename(columns=columns_dict)
    output_valid_df = output_valid_df[columns_dict.values()]
    output_valid_df.to_csv(PATH_CSV, index=False)
    
    # 评估
    from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
    print("Classification Report:")
    y_class_test = valid_df.next_based_index_class
    y_class_pred = valid_df.next_based_index_class_pred
    conf_matrix = confusion_matrix(y_class_test,
                                   y_class_pred,
                                   labels=[-2, -1, 0, 1, 2])
    print(conf_matrix)
    class_report_df = pd.DataFrame(classification_report(y_class_test, y_class_pred, output_dict=True)).T
    class_report_df=class_report_df.head(-3) # del 'accuracy', 'macro avg', 'weighted avg'
    fbeta = fbeta_score(y_class_test, y_class_pred, beta=0.5, average=None)
    class_report_df['f05-score'] = fbeta
    print(class_report_df)