# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
测试(test_main)
"""
import os
import argparse

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log
#from tests_ import test_0_lightgbm

from tests_ import test_2_class_base_index, test_2_reg_price

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

PATH_CSV = f'{PATH}/data/test_main.csv'


class TestMainLightgbm:
    def __init__(self):
        self.test_class_base_index = test_2_class_base_index.TestClassBaseIndex()
        self.test_price = test_2_reg_price.TestPrice()
        
    def pipeline(self, df):
        class_base_index_df = self.test_class_base_index.test_board_pipline(asset_df)
        price_df = self.test_price.test_board_pipline(asset_df)
        
        # class_base_index_df = class_base_index_df.drop_duplicates('primary_key',keep='first')
        # price_df = price_df.drop_duplicates('primary_key',keep='first')
        
        #df = pd.merge(df, class_base_index_df, on='primary_key')
        prediction_df = pd.merge(class_base_index_df, price_df, on='primary_key')
        return prediction_df

def calculate_next_close(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    df[['next_close']] = df[['close']].shift(-1)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2024-12-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2025-01-10', help='End time for backtesting')
    args = parser.parse_args()

    logger.info(f"""
    task: test_2_reg_price_high
    date_start: {args.date_start}
    date_end: {args.date_end}
    """)
    
    # dataset
    with utils_database.engine_conn("POSTGRES") as conn:
        asset_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    asset_df.drop_duplicates('primary_key', keep='first', inplace=True)
    
    test_main_lightgbm = TestMainLightgbm()
    valid_raw_df = test_main_lightgbm.pipeline(asset_df)
    
    # output
    valid_df = pd.merge(valid_raw_df, asset_df[['primary_key','next_date','next_high_rate','next_low_rate','next_close_rate','next_based_index_class','board_type','price_limit_rate','open','high',
                                                'low','close','volume','turnover','turnover_pct','chg_rel','date','full_code','code_name']], how='left', on='primary_key')
    
    valid_df = valid_df.groupby('full_code').apply(calculate_next_close)
    
    valid_df['next_high'] = valid_df['next_high_rate'] * valid_df['close']
    valid_df['next_high_pred'] = valid_df['next_high_rate_pred'] * valid_df['close']
    valid_df['next_low'] = valid_df['next_low_rate'] * valid_df['close']
    valid_df['next_low_pred'] = valid_df['next_low_rate_pred'] * valid_df['close']    
    valid_df['next_close'] = valid_df['next_close_rate'] * valid_df['close']
    valid_df['next_close_pred'] = valid_df['next_close_rate_pred'] * valid_df['close']
    
    # valid_df[['next_high','next_low','next_high_pred','next_low_pred']] = valid_df[['next_high','next_low','next_high_pred','next_low_pred']].round(2)
    # valid_df[['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred']] = valid_df[['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred']].round(4)
    
    columns_round_2 = ['high','low','close','next_high','next_low','next_close','next_high_pred','next_low_pred','next_close_pred']
    valid_df[columns_round_2] = valid_df[columns_round_2].round(2)
    
    columns_round_4 = ['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred','next_close_rate','next_close_rate_pred']
    valid_df[columns_round_4] = valid_df[columns_round_4].round(4)

    columns_dict = {'date': '数据日期',
                    'next_date': '预测日期',
                    'full_code': '股票代码',
                    'code_name': '公司名称',
                     'board_type': '板块',
                    #'price_limit_rate': '涨跌停比例',
                    #'open': '开盘价',
                     'high': '最高价',
                     'low': '最低价',
                     'close': '收盘价',
                     #'volume': '成交数量',
                     #'turnover': '成交金额',
                     #'turnover_pct': '换手率',
                     #'chg_rel': '涨跌幅',
                     'next_low': '明天_最低价_真实值',
                     'next_low_pred': '明天_最低价_预测值',
                     #'next_low_rate': '明天_最低价幅_真实值',
                     #'next_low_rate_pred': '明天_最低价幅_预测值',
                     'next_high': '明天_最高价_真实值',
                     'next_high_pred': '明天_最高价_预测值',
                     #'next_high_rate': '明天_最高价幅_真实值',
                     #'next_high_rate_pred': '明天_最高价幅_预测值',
                     'next_close': '明天_收盘价',
                     'next_based_index_class': '明天_基于指数分类_真实值',
                     'next_based_index_class_pred': '明天_基于指数分类_预测值',
                     #'base_index_-1_prob': '明天_基于指数-1_概率',
                     #'base_index_0_prob': '明天_基于指数0_概率',
                     #'base_index_1_prob': '明天_基于指数1_概率',
                    }
    valid_df['rank_score'] = valid_df['base_index_-1_prob']*0 +\
                            valid_df['base_index_0_prob']*1 +\
                            valid_df['base_index_1_prob']*2 +\
                            valid_df['next_close_rate_pred']
                        
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
    #prediction_df.to_sql('rl_environment', con=conn.engine, index=False, if_exists='append')
