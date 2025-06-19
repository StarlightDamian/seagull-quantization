# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:36:50 2024

@author: awei
偏置模型示例(demo_train_2_bias)
"""
import lightgbm as lgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from __init__ import path

import numpy as np

# 自定义损失函数
def custom_loss(y_pred, dataset):
    y_true = dataset.get_label()
    data_type = dataset.params.get('data_type', '')

    # 根据数据类型确定 next_close_real 值
    #if data_type == 'train':
    next_close_real = X_train['next_close_real']
    
    # 设定偏差数组
    bias_array = np.arange(-0.5, 0.51, 0.003)
    y_pred_matrix = y_pred[:, np.newaxis] + bias_array

    # 计算收益矩阵
    reward_matrix = np.where(
        y_pred_matrix <= y_true[:, np.newaxis],
        y_pred_matrix - 1,
        np.tile((next_close_real - 1).values[:, np.newaxis], (1, len(bias_array)))
    )
    
    # 计算平均收益和最佳偏差
    reward_mean = np.mean(reward_matrix, axis=0)
    max_index = np.argmax(reward_mean)
    best_bias = bias_array[max_index]
    
    # 使用最佳偏差调整预测值
    adjusted_y_pred = y_pred + best_bias

    # 计算梯度和 Hessian
    residual = (y_true - adjusted_y_pred).astype("float")
    grad = np.where(
        residual >= 0,
        -2 * residual,  # 如果预测值小于真实值，则增大预测值
        2 * residual    # 如果预测值大于真实值，则减小预测值
    )
    hess = np.ones_like(grad) * 2  # 这里假设二阶导数常数为2
    
    return grad, hess


# 自定义评估指标函数
def custom_metric(y_pred, dataset):
    y_true = dataset.get_label()
    data_type=dataset.params.get('data_type', '')
    if data_type == 'train':
        # 处理训练集的逻辑
        next_close_real = X_train.next_close_real
    elif data_type == 'valid':
        # 处理验证集的逻辑
        next_close_real = X_test.next_close_real
    
    # reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
    # total_profit = np.prod(reward, axis=0)**0.5
    # total_profit = np.mean(reward) 
    bias_array = np.arange(-0.5, 0.51, 0.003)
    y_pred_matrix = y_pred[:, np.newaxis] + bias_array
    
# =============================================================================
#     if data_type == 'train':
#         train_results = column_mean
#     elif data_type == 'valid':
#         valid_results = column_mean
# =============================================================================
    reward_matrix = np.where(y_pred_matrix <= y_true[:, np.newaxis],
                           y_pred_matrix - 1,
                           np.tile((next_close_real - 1).values[:, np.newaxis], (1, len(bias_array)))
                           )    

    reward_mean = np.mean(reward_matrix, axis=0)
    max_index = np.argmax(reward_mean, axis=0)
    print(f'{data_type}: {max_index}| {bias_array[max_index]:.5f}')
    global best_train_bias
    if data_type == 'train':
        best_train_bias = bias_array[max_index]
    total_profit = reward_mean[max_index]
    
    # 计算其他指标
    valid_preds = np.sum(y_pred <= y_true)
    total_preds = len(y_pred)
    valid_ratio = valid_preds / total_preds
    #print(f"Valid predictions ratio: {valid_ratio:.4f} ({valid_preds}/{total_preds})")

    return '|', total_profit, True  # 名称, 值, 是否越大越好

#valid_custom_profit:0.00381579
#rmse: 0.03992686340357715
# LightGBM 参数
params = {
    'objective': 'regression',#'regression',custom_loss
    'metric': 'rmse',#'custom',#'custom', # rmse,mae# 使用自定义损失函数时，需要设置metric为['None','custom']
    'verbose': -1,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'max_depth': -1,
}
# =============================================================================
# params = {
#     'boosting_type': 'gbdt',
#     'objective': custom_loss,
#     'metric': 'None',  # 使用自定义损失函数时，需要设置metric为'None'
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': -1
# }
# =============================================================================
# 生成示例数据
data = pd.read_csv(f'{path}/data/test_603893.csv')
#data['high'] = data['high'] / data['close']
columns_to_divide = ['high', 'low', 'open', 'close']
data[['high_pct', 'low_pct', 'open_pct', 'close_pct']] = data[columns_to_divide].div(data['preclose'], axis=0)

data[['next_high', 'next_low','next_open']] = data[['high', 'low','open']].shift(-1)
data['next_close_real'] = data['close'].shift(-1)
data = data.head(-1)
feature_name=['open', 'high', 'low', 'close', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
       'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','next_open','next_close_real']
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
lgb_train = lgb.Dataset(X_train,
                        label=y_train,
                        # feature_name=feature_name,
                        free_raw_data=False)
lgb_valid = lgb.Dataset(X_test,
                        label=y_test,
                        reference=lgb_train,
                        # feature_name=feature_name,
                        free_raw_data=False)

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

# 训练模型
model = lgb.train(
    params=params,
    train_set=lgb_train_1,
    valid_sets=[lgb_train_1, lgb_valid_1],
    feval=custom_metric,
    num_boost_round=100,
    callbacks=[lgb.callback.log_evaluation(period=10)]
)
y_pred = model.predict(X_test_1)
#y_pred+=best_train_bias
# 计算 MSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"rmse: {rmse}")

result = pd.DataFrame([y_test.values, y_pred]).T
result.columns = ['y_test','y_pred']
print(result)
result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

result.to_csv(f'{path}/data/test_result_reward2.csv',index=False)

result_bool = result[result.next_high_bool==1]
y_test,y_pred,next_high_bool = result_bool.mean()
print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])

reward = (y_pred-1)*(result_bool.shape[0])
reward_all = (y_test-1)*(result.shape[0])
reward_pct = reward/reward_all
print(f'reward_pct: {reward_pct:.4f}')
# 训练模型，使用自定义损失函数
#y_test            1.030229
#y_pred             1.01748
# 137

# RMSE
#y_test            1.051216
#y_pred            1.025275
#105
#reward_pct: 0.2479