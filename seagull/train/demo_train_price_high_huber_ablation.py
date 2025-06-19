# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 18:02:46 2024

@author: awei
最高价卖出价格模型消融实验示例(demo_train_price_high_huber_ablation)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import HuberRegressor

from seagull.settings import PATH
from seagull.utils import utils_database


# 定义自定义Huber损失
def custom_huber_loss(y_pred, dataset):
    y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数，可以调整

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, -alpha * np.sign(delta))
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)
    
    return grad, hess

def dataset():
    with utils_database.engine_conn("POSTGRES") as conn:
        # dwd_freq_incr_stock_daily
        data = pd.read_sql("select * from ods_freq_incr_baostock_stock_sh_sz_daily where date BETWEEN '2022-01-01' AND '2023-01-01'and code='sh.601166'", con=conn.engine)
        #data[['high', 'low', 'open', 'close','preclose']] = data[['high', 'low', 'open', 'close','preclose']].astype(float)
        
        
    #data = pd.read_csv(f'{PATH}/_file/test_603893.csv')
    #data['high'] = data['high'] / data['close']
    columns_to_divide = ['high', 'low', 'open', 'close']
    data[['open', 'high', 'low', 'close', 'volume',
           'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
           'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','preclose']] = data[['open', 'high', 'low', 'close', 'volume',
                  'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
                  'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','preclose']].astype(float)
    
    data[columns_to_divide] = data[columns_to_divide].div(data['preclose'], axis=0)

    data[['next_high', 'next_low']] = data[['high', 'low']].shift(-1)
    data['next_close_real'] = data['close'].shift(-1)
    data = data.head(-1)
    feature_name=['open', 'high', 'low', 'close', 'volume',
           'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
           'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']#,'next_close_real'
    X = data[feature_name]
    #y = data[['next_high', 'next_low']]
    y = data['next_high']
    # 拆分数据集
    # 假设 X 和 y 已经准备好
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_name_1=['open', 'high', 'low', 'close', 'volume',
           'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
           'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']
    X_train_1 = X_train[feature_name_1]
    X_test_1 = X_test[feature_name_1]

    # 创建 Dataset 时包含所有特征
    lgb_train_1 = lgb.Dataset(X_train_1,
                            label=y_train,
                            # feature_name=feature_name,
                            free_raw_data=False,
                            params={'data_type': 'train'},
                            )
    lgb_valid_1 = lgb.Dataset(X_test_1,
                            label=y_test,
                            reference=lgb_train_1,
                            # feature_name=feature_name,
                            free_raw_data=False,
                            params={'data_type': 'valid'},
                            )
    return X_train, X_test, y_train, y_test

def fit_model(X_train, X_test, y_train, y_test, params):
    """
    Fits the model to single or multi-output training data.
    
    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Target values. Can be 1D (Series) or 2D (DataFrame/ndarray).
        
    Returns:
        Fitted model.
    """
    
    if isinstance(y_train, pd.Series) or y_train.ndim == 1: # Single-output case
        #model = lgb_regressor
        
        lgb_train_1 = lgb.Dataset(X_train,
                                label=y_train,
                                # feature_name=feature_name,
                                free_raw_data=False,
                                params={'data_type': 'train'},
                                )
        lgb_valid_1 = lgb.Dataset(X_test,
                                label=y_test,
                                reference=lgb_train_1,
                                # feature_name=feature_name,
                                free_raw_data=False,
                                params={'data_type': 'valid'},
                                )
        # Fit the model
        model = lgb.train(
            params=params,
            train_set=lgb_train_1,
            valid_sets=[lgb_train_1, lgb_valid_1],
            num_boost_round=100
        )
    else:# Multi-output case
        lgb_regressor = lgb.LGBMRegressor(**params)
        model = MultiOutputRegressor(lgb_regressor)
        # Fit the model
        model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test  = dataset()

    # 定义参数
    params = {
    'objective': custom_huber_loss,#'huber',  # 直接使用LightGBM内置的Huber损失
    'alpha': 1.2,  # 可以调整Huber损失的阈值参数
    'metric': 'None',
    'learning_rate': 0.05,
    'num_leaves': 62,
    'min_data_in_leaf': 20,
    'max_depth': -1,
}
# =============================================================================
#     params = {
#         'objective': custom_huber_loss,  # 使用自定义Huber损失
#         'metric': 'None',  # 需要设置metric为None
#         'learning_rate': 0.05,
#         'num_leaves': 62,
#         'min_data_in_leaf': 20,
#         'max_depth': -1,
#     }
# =============================================================================
    
    # model = MultiOutputRegressor(lgb_regressor)
    #model.fit(X_train, y_train)
    

    model = fit_model(X_train, X_test, y_train, y_test, params)

    # 训练模型
# =============================================================================
#     model = lgb.train(
#         params=params,
#         train_set=lgb_train_1,
#         valid_sets=[lgb_train_1, lgb_valid_1],
#         num_boost_round=100
#     )
# =============================================================================

    # 预测
    y_pred = model.predict(X_test)
    y_pred = y_pred - 0.000009  # 要预测的实际上是y_pred小于y_true的部分

    result = pd.DataFrame([y_test.values, y_pred]).T
    result.columns = ['y_test','y_pred']
    print(result)
    result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

    result.to_csv(f'{PATH}/_file/test_result_reward2.csv',index=False)

    result_bool = result[result.next_high_bool==1]
    y_test_mean, y_pred_mean,next_high_bool = result_bool.mean()
    print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])

    reward = (y_pred_mean-1)*(result_bool.shape[0])
    reward_all = (y_test_mean-1)*(result.shape[0])
    reward_pct = reward/reward_all
    print(f'reward_pct: {reward_pct:.4f}')