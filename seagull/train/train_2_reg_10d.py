# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 00:29:56 2024

@author: awei
未来十日价格模型(train_2_reg_10d)
建议
学习率调优：根据损失函数的定义，调优模型的学习率。过高的学习率可能导致模型不收敛，过低的学习率可能导致学习过慢或停滞。
初始化 y_pred：确保模型初始化时 y_pred 不全是 0，某些模型参数或训练过程可能会导致初始预测值过于单一。
验证数据集结构：确保 next_close_real 数据正确加载，并与 y_true 及 y_pred 保持一致。

损失函数修改为最大化收益的方式。其原理是这样，一共三列['y_true ','y_pred','next_close_real']，
每次的1是基准数据，猜测y_true 最大是多少，并且y_pred尽量小于y_true ，如果y_true >=y_pred,则reward为y_pred-1，。如果y_true <y_pred,则reward为next_close_real

if y_pred <= y_true:
    reward = y_pred-1
elif y_pred > y_true:
    reward = next_close_real-1

Mean Squared Error: 0.0007571028322268891
"""
import os
import argparse

import numpy as np
import pandas as pd


from seagull.settings import PATH
from train import train_1_lightgbm_regression
from feature import feature_engineering_main
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

TARGET_NAMES = ['next_high_rate',
                'next_low_rate',
                'next_close_rate',
                'y_10d_vwap_rate',
                'y_10d_max_dd',
                'y_10d_high_rate',
                'y_10d_low_rate',
                ]

PATH_CSV = f'{PATH}/_file/train_price.csv'


class TrainPrice(train_1_lightgbm_regression.LightgbmRegressionTrain):
    def __init__(self):
        super().__init__()
        self.target_names = TARGET_NAMES
        
        ## train_model
        self.task_name = 'reg_price'
        
        self.params = {'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'max_depth': 9,
                'num_leaves': 511,  # 决策树上的叶子节点的数量，控制树的复杂度
                'learning_rate': 0.12,  # 0.05,0.1
                'metric': ['root_mean_squared_error'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
                #w×RMSE+(1−w)×MAE
                'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
                #'early_stop_round':20,
                'n_estimators': 1500,
                #'min_child_sample':40,
                #'min_child_weight':1,
                #'subsample':0.8,
                #'colsample_bytree':0.8,
    }
    # 自定义收益最大化损失函数
    def custom_loss(self, y_pred, dataset):
        y_true = dataset.get_label()
        delta = y_true - y_pred
        alpha = 1.2  # Huber 损失的阈值参数，可以调整
        
        # 梯度和 Hessian
        grad = np.where(np.abs(delta) <= alpha, -delta, -alpha * np.sign(delta))
        hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)
        
        return grad, hess
    
    def custom_metric(self, y_pred, dataset):
        # 自定义评估指标函数
        y_true = dataset.get_label()
        next_close_real = dataset.get_data()['next_close_real']
        reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
        # total_profit = np.sum(reward)
        total_profit = np.prod(reward, axis=0)**0.5
        return 'custom_profit', total_profit, True  # 名称, 值, 是否越大越好


if __name__ == '__main__':
    # date in 1990-12-19, 2024-08-14
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2014-01-01', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2023-04-01', help='Start time for training')
    #parser.add_argument('--date_end', type=str, default='2023-12-01', help='end time of training')
    parser.add_argument('--date_start', type=str, default='2023-01-03', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2024-10-03', help='Start time for training')
    #parser.add_argument('--date_end', type=str, default='2024-11-01', help='end time of training')
    parser.add_argument('--date_end', type=str, default='2024-12-20', help='end time of training')
    args = parser.parse_args()
    
    logger.info(f"""
    task: train_2_reg_price_high
    date_start: {args.date_start}
    date_end: {args.date_end}
    """)
    
    # dataset
    with utils_database.engine_conn("POSTGRES") as conn:
        asset_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    asset_df.drop_duplicates('primary_key', keep='first', inplace=True)
    
    # 清洗脏数据
    asset_df = asset_df[(asset_df.high <= asset_df.limit_up)&
                        (asset_df.low >= asset_df.limit_down)&
                        (asset_df.next_high_rate-1<asset_df.price_limit_rate)&
                        (1-asset_df.next_low_rate<asset_df.price_limit_rate)
                        ]
    asset_df = asset_df[~((asset_df.next_high_rate.apply(np.isinf))|
                          (asset_df.next_low_rate.apply(np.isinf))|
                          (asset_df.y_10d_vwap_rate.apply(np.isinf))|
                          (asset_df.y_10d_max_dd.apply(np.isinf))
                          )]
    
    # 去除涨停
    asset_df = asset_df[~((asset_df.is_limit_up==True)|(asset_df.is_limit_down==True))]
    # asset_df.loc[asset_df.next_high_rate.idxmax(),['close','high','next_high_rate','price_limit_rate']]
    
    asset_df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
    #asset_df.reset_index(drop=True).to_feather(f'{PATH}/_file/das_wide_incr_train_20230103_20241220.feather')
    train_price = TrainPrice()
    valid_raw_df = train_price.train_board_pipeline(asset_df, keep_train_model=True)
    
    ## output
    valid_df = pd.merge(valid_raw_df,
                        asset_df[['primary_key','next_high_rate','next_low_rate','y_10d_low_rate','y_10d_high_rate',
                                  'y_10d_vwap_rate','y_10d_max_dd','price_limit_rate','open','high','low',
                                  'close','volume','turnover','turnover_pct','chg_rel','date','full_code','code_name']],
                        how='left',
                        on='primary_key')
    
    # 最高价
    valid_df['next_high'] = valid_df['next_high_rate'] * valid_df['close']
    valid_df['next_high_pred'] = valid_df['next_high_rate_pred'] * valid_df['close']
    
    # 最低价
    valid_df['next_low'] = valid_df['next_low_rate'] * valid_df['close']
    valid_df['next_low_pred'] = valid_df['next_low_rate_pred'] * valid_df['close']
    
    # 10日vwap
    valid_df['y_10d_vwap'] = valid_df['y_10d_vwap_rate'] * valid_df['close']
    valid_df['y_10d_vwap_pred'] = valid_df['y_10d_vwap_rate_pred'] * valid_df['close']
    
    # 10日回撤
    # valid_df['y_10d_max_dd_pred'] = valid_df['y_10d_max_dd'] * valid_df['close']
    # valid_df['y_10d_max_dd_pred'] = valid_df['y_10d_max_dd_pred'] * valid_df['close']
    
    # 10日vwap回撤比
    valid_df['y_10d_vwap_drawdown_rate'] = (valid_df['y_10d_vwap_rate_pred'] / (valid_df['y_10d_max_dd_pred'] + 1))
    valid_df['y_10d_vwap_drawdown_pct'] = valid_df['y_10d_vwap_drawdown_rate'] * 100
    
    # 10日最低价
    valid_df['y_10d_low'] = valid_df['y_10d_low_rate'] * valid_df['close']
    valid_df['y_10d_low_pred'] = valid_df['y_10d_low_rate_pred'] * valid_df['close']
    
    # 10日最高价
    valid_df['y_10d_high'] = valid_df['y_10d_high_rate'] * valid_df['close']
    valid_df['y_10d_high_pred'] = valid_df['y_10d_high_rate_pred'] * valid_df['close']
    
    # 小数位数
    round_2_columns = ['y_10d_vwap_drawdown_pct']
    valid_df[round_2_columns] = valid_df[round_2_columns].round(2)
    
    round_3_columns = ['next_high','next_low','next_high_pred','next_low_pred','y_10d_vwap','y_10d_vwap_pred',
                       'y_10d_low','y_10d_low_pred','y_10d_high','y_10d_high_pred']
    valid_df[round_3_columns] = valid_df[round_3_columns].round(3)
    
    round_4_columns = ['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred','y_10d_max_dd',
                       'y_10d_max_dd_pred']
    valid_df[round_4_columns] = valid_df[round_4_columns].round(4)
    
    valid_df = valid_df.sort_values(by='y_10d_vwap_drawdown_pct' ,ascending=False)
    
    columns_dict = {'date': '日期',
                    'full_code': '股票代码',
                    'code_name': '公司名称',
                    'open': '开盘价',
                    'high': '最高价',
                    'low': '最低价',
                    'close': '收盘价',
                    'volume': '成交数量',
                    'turnover': '成交金额',
                    'turnover_pct': '换手率',
                    'next_low': '明天_最低价_真实值',
                    'next_low_pred': '明天_最低价_预测值',
                    'next_low_rate': '明天_最低价幅_真实值',
                    'next_low_rate_pred': '明天_最低价幅_预测值',
                    'next_high': '明天_最高价_真实值',
                    'next_high_pred': '明天_最高价_预测值',
                    'next_high_rate': '明天_最高价幅_真实值',
                    'next_high_rate_pred': '明天_最高价幅_预测值',
                    'y_10d_vwap': '10日_平均成本_真实值',
                    'y_10d_vwap_pred': '10日_平均成本_预测值',
                    'y_10d_max_dd': '10日_回撤_真实值',
                    'y_10d_max_dd_pred': '10日_回撤_预测值',
                    'y_10d_low': '10日_最低价_真实值',
                    'y_10d_low_pred': '10日_最低价_预测值',
                    'y_10d_high': '10日_最高价_真实值',
                    'y_10d_high_pred': '10日_最高价_预测值',
                    'y_10d_vwap_drawdown_pct': '单位风险收益_预测值',
                    'price_limit_rate': '涨跌停比例',
                    #'macro_value_traded': '深沪成交额',
                    #'macro_value_traded_diff_1': '深沪成交额增量',
                    #'chg_rel': '涨跌幅',
                    #'next_close_rate_pred',
                    #'y_10d_vwap_rate_pred',
                    #'y_10d_high_rate_pred',
                    #'y_10d_low_rate_pred',
                    #'primary_key',
                    #'y_10d_low_rate',
                    #'y_10d_high_rate',
                    #'y_10d_vwap_rate',
                    }
    output_valid_df = valid_df.rename(columns=columns_dict)
    output_valid_df = output_valid_df[columns_dict.values()]
    output_valid_df.to_csv(PATH_CSV, index=False)
    
    # valid_df[['y_10d_vwap_rate_pred','y_10d_max_dd_pred','y_10d_vwap_drawdown_pct']]
    # valid_df.loc[valid_df['y_10d_vwap_drawdown_pct'].idxmax()]
    
