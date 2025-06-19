# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
测试基于指数分类模型(test_2_class_base_index)
"""
import os
import argparse

import pandas as pd

from __init__ import path
from utils import utils_database, utils_log
from tests_ import test_0_lightgbm

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


TASK_NAME = 'class_base_index'
TEST_TABLE_NAME = 'test_2_class_base_index'
PATH_CSV = f'{path}/data/test_2_class_base_index.csv'

class TestClassBaseIndex(test_0_lightgbm.lightgbmTest):
    def __init__(self, multioutput_model_path=None):
        super().__init__()
        self.test_table_name = TEST_TABLE_NAME
        #self.multioutput_model_path = multioutput_model_path#MULTIOUTPUT_MODEL_PATH
        self.task_name = TASK_NAME
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-12-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for backtesting')
    args = parser.parse_args()

    logger.info(f"""
    task: test_2_reg_price_high
    date_start: {args.date_start}
    date_end: {args.date_end}
    """)
    
    # dataset
    with utils_database.engine_conn('postgre') as conn:
        asset_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    asset_df.drop_duplicates('primary_key', keep='first', inplace=True)
    
    
    test_class_base_index = TestClassBaseIndex()
    valid_raw_df = test_class_base_index.test_board_pipline(asset_df)
    
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