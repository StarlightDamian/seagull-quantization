# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:05:25 2024

@author: awei
最高价卖出价格模型示例(demo_train_price_high_huber)
epsilon: 增大它会使模型更像普通线性回归，减小它会使模型更像 LAD(最小绝对偏差)
alpha: 增大它会增加正则化强度
max_iter: 如果模型没有收敛，可以增加这个值
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from seagull.settings import PATH

# 定义自定义Huber损失
# =============================================================================
# def custom_huber_loss(y_pred, dataset):
#     y_true = dataset.get_label()
#     delta = y_true - y_pred
#     print('delta',delta)
#     global delta1
#     delta1 = delta
#     alpha = 1.2  # Huber 损失的阈值参数，可以调整
#     k1 = 0.6  # 左侧斜率系数
#     k2 = 1.5  # 右侧斜率系数
# 
#     # 梯度和 Hessian
#     grad = np.where(np.abs(delta) <= alpha, -delta, 
#                     np.where(delta < 0,
#                              -k1 * -alpha * np.sign(delta),
#                              -k2 * -alpha * np.sign(delta)))  # 根据delta值选择不同的斜率
#     hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)  # 斜率保持不变（或可根据需要调整）
# 
#     return grad, hess
# =============================================================================
def custom_huber_loss(y_pred, dataset):
    y_true = dataset.get_label()
    delta = y_true - y_pred
    threshold_grad = 1.2
    threshold_hess = 1.2
    alpha = 1.2  # Huber 损失的阈值参数，可以调整

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= threshold_grad, -delta, -alpha * np.sign(delta))
    hess = np.where(np.abs(delta) <= threshold_hess, 0.9, 0.2)
    
    return grad, hess

# =============================================================================
# # huber 损失
# def custom_huber_loss(y_pred, dataset):
#     y_true = dataset.get_label()
#     delta = 0.05#1.25 # 设置 delta 阈值
#     error = y_true - y_pred
#     # 计算 loss
#     #loss = np.where(np.abs(error) < delta,
#     #                0.5 * (error)**2,
#     #                delta * np.abs(error) - 0.5 * (delta**2))
# 
#     k1 = 1.3  # 左侧斜率系数
#     k2 = 1.115  # 右侧斜率系数
#     # 计算梯度
#     grad = np.where(np.abs(error) < delta,
#                     -error,
#                     #-delta * np.sign(error),
#                     np.where(error < 0,
#                              -k1 * error,
#                              -k2 * error)
#                     )
# 
#     # 计算 Hessian
#     hess = np.where(np.abs(error) < delta,
#                     0.9,#1,  # 二次损失部分，Hessian 为 1
#                     0#.2#0,
#                     )  # 线性损失部分，Hessian 为 0
# 
#     return grad, hess
# =============================================================================
# =============================================================================
# def custom_huber_loss(y_pred, dataset):
#     y_true = dataset.get_label()
#     delta=1
#     loss = np.where(np.abs(y_true-y_pred)<delta,
#                     0.5*((y_true-y_pred)**2),
#                     delta*np.abs(y_true - y_pred)-0.5*(delta**2))
#     hess = np.where(np.abs(y_true - y_pred) <= 1.2, 0.9, 0.2)
#     return np.sum(loss), hess
# 
# =============================================================================
# 定义数据集
# 生成示例数据
data = pd.read_csv(f'{PATH}/_file/test_603893.csv')
#data['high'] = data['high'] / data['close']
columns_to_divide = ['high', 'low', 'open', 'close']
data[columns_to_divide] = data[columns_to_divide].div(data['preclose'], axis=0)

data[['next_high', 'next_low']] = data[['high', 'low']].shift(-1)
data['next_close_real'] = data['close'].shift(-1)
data = data.head(-1)
feature_name=['open', 'high', 'low', 'close', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
       'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','next_close_real']
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

# 定义参数
params = {
    'objective': custom_huber_loss,  # 使用自定义Huber损失
    #'metric': 'None',  # 需要设置metric为None
    'learning_rate': 0.05,
    'num_leaves': 62,
    'min_data_in_leaf': 20,
    'max_depth': -1,
}

# 训练模型
model = lgb.train(
    params=params,
    train_set=lgb_train_1,
    valid_sets=[lgb_train_1, lgb_valid_1],
    num_boost_round=100
)

# 预测
y_pred = model.predict(X_test_1)
y_pred = y_pred-0.000009  # 要预测的实际上是y_pred小于y_true的部分

result = pd.DataFrame([y_test.values, y_pred]).T
result.columns = ['y_test','y_pred']
print(result)
result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

result.to_csv(f'{PATH}/_file/test_result_reward2.csv',index=False)

result_bool = result[result.next_high_bool==1]
y_test_mean, y_pred_mean, next_high_bool = result_bool.mean()
print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])

reward = (y_pred_mean-1)*(result_bool.shape[0])
reward_all = (y_test_mean-1)*(result.shape[0])
reward_pct = reward/reward_all
print(f'reward_pct: {reward_pct:.4f}')

# =============================================================================
# y_test            1.046151
# y_pred            1.021096
# next_high_bool         1.0
# dtype: object 124 / 209
# reward_pct: 0.2712
# =============================================================================
# =============================================================================
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import make_scorer
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# from scipy.special import huber
# 
# # 定义自定义的 Huber 损失函数
# def huber_loss(y_true, y_pred, delta=1.0):
#     error = y_true - y_pred
#     is_small_error = np.abs(error) <= delta
#     squared_loss = 0.5 * error ** 2
#     linear_loss = delta * (np.abs(error) - 0.5 * delta)
#     return np.where(is_small_error, squared_loss, linear_loss)
# 
# # 定义评分函数，以便在模型评估中使用
# huber_scorer = make_scorer(lambda y_true, y_pred: np.mean(huber_loss(y_true, y_pred)), greater_is_better=False)
# 
# # 生成示例数据
# X = np.random.rand(100, 2) * 10
# y = 3 * np.sin(X[:, 0]) + np.random.randn(100) * 0.5  # 使用非线性关系
# 
# # 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # 使用非线性回归模型（例如，梯度提升回归器）
# model = GradientBoostingRegressor(loss='huber', alpha=0.9)  # alpha定义Huber分位数
# model.fit(X_train, y_train)
# 
# # 预测
# y_pred = model.predict(X_test)
# 
# # 使用 Huber 损失函数进行评估
# huber_loss_value = np.mean(huber_loss(y_test, y_pred))
# print("Huber Loss:", huber_loss_value)
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# =============================================================================


