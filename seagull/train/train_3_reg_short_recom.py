# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:01:48 2024

@author: awei
训练短期推荐模型(train_3_reg_short_recom)
"""
import argparse
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from seagull.settings import PATH
from train import train_1_lightgbm_regression
from base import base_connect_database, base_trading_day, base_utils

#EVAL_TABLE_NAME = 'eval_train'
TRAIN_TABLE_NAME = 'train_3_short_term_recommend'
TARGET_REAL_NAMES = ['rear_next_rise_pct_real', 'rear_next_fall_pct_real', 'rear_next_pct_real']
#MULTIOUTPUT_MODEL_PATH = f'{PATH}/checkpoint/lightgbm_regression_short_term_recommend.joblib'
PREDICTION_PRICE_OUTPUT_CSV_PATH = f'{PATH}/_file/train_short_term_recommend.csv'


class trainShortTermRecommend(train_1_lightgbm_regression.lightgbmRegressionTrain):
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
        
def feature_engineering_short_term_recommend(date_start, date_end):
    # Load date range data
    with base_connect_database.engine_conn("POSTGRES") as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{date_start}' AND date < '{date_end}'", con=conn.engine)
        
        test_stock_pick_df = pd.read_sql(f"SELECT * FROM test_1_stock_pick WHERE date >= '{date_start}' AND date < '{date_end}'", con=conn.engine)
    
    history_day_df.drop_duplicates('primary_key',keep='first', inplace=True)
    test_stock_pick_df.drop_duplicates('primary_key',keep='first', inplace=True)
    
    history_day_df = pd.merge(history_day_df, test_stock_pick_df[['primary_key','rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'rear_open_pct_pred', 'rear_close_pct_pred']], on='primary_key')
    
    trading_day = base_trading_day.tradingDay()
    trading_day_after_dict = trading_day.specified_trading_day_after()
    history_day_df['date_after_1'] = history_day_df['date'].map(trading_day_after_dict)
    history_day_df['date_after_2'] = history_day_df['date_after_1'].map(trading_day_after_dict)
    
    after_1_df = history_day_df[['date_after_1','code','low','high']]
    after_1_df['primary_key'] = (after_1_df['date_after_1']+after_1_df['code']).apply(base_utils.md5_str)
    after_1_df = after_1_df.rename(columns={'low': 'rear_low',
                                            'high': 'rear_high'})
    
    after_2_df = history_day_df[['date_after_2','code','low','high']]
    after_2_df['primary_key'] = (after_2_df['date_after_2']+after_1_df['code']).apply(base_utils.md5_str)
    after_2_df = after_2_df.rename(columns={'low':'rear_2_low',
                                            'high': 'rear_2_high'})
    
    history_day_df = pd.merge(history_day_df, after_1_df[['primary_key', 'rear_low', 'rear_high']], on='primary_key')
    history_day_df = pd.merge(history_day_df, after_2_df[['primary_key', 'rear_2_low', 'rear_2_high']], on='primary_key')
    
    history_day_df['rear_next_rise_pct_real'] = ((history_day_df['rear_2_high'] - history_day_df['rear_low']) / history_day_df['rear_low']) * 100
    
    history_day_df['rear_next_fall_pct_real'] = ((history_day_df['rear_high'] - history_day_df['rear_2_low']) / history_day_df['rear_2_low']) * 100
    
    
    history_day_df['rear_next_pct_real'] = (history_day_df['rear_2_high'] * history_day_df['rear_2_low'] - history_day_df['rear_high'] * history_day_df['rear_low']) * 100 / (history_day_df['rear_low'] * history_day_df['rear_2_low'])
    
    history_day_df.drop(columns=['date_after_1', 'date_after_2', 'rear_low','rear_high','rear_2_low', 'rear_2_high'], inplace=True)
    #print(history_day_df.shape[0])
    #print(history_day_df.columns)
    #['primary_key', 'date', 'code', 'code_name', 'open', 'high', 'low','close', 'preclose', 'volume', 'amount', 'adjustflag', 'turn','tradestatus', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred','rear_open_pct_pred', 'rear_close_pct_pred', 'rear_next_rise_pct_real']
    return history_day_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2022-06-01', help='Start time of training')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='end time of training')
    #parser.add_argument('--date_end', type=str, default='2024-03-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # 根据数据库该表test_stock_pick的日期进行二次训练
    history_day_short_term_df = feature_engineering_short_term_recommend(args.date_start, args.date_end)
    
    train_stock_term_recommend = trainShortTermRecommend()
    #train_stock_term_recommend.grid_search(history_day_short_term_df)
    prediction_df = train_stock_term_recommend.train_pipline(history_day_short_term_df)
    
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
