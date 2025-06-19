# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:36:50 2024

@author: awei
最高价卖出价格模型示例(demo_train_2_stock_price_high)
其他建议
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
import lightgbm as lgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from seagull.settings import PATH
def custom_loss( y_pred, dataset):
    y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数，可以调整
    k1 = 1.0  # 左侧斜率系数
    k2 = 1.5  # 右侧斜率系数

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, 
                    np.where(delta < 0, -k1 * delta, -k2 * delta))  # 根据delta值选择不同的斜率
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)  # 斜率保持不变（或可根据需要调整）

    return grad, hess
# 自定义收益最大化损失函数
# =============================================================================
# def custom_loss(y_pred, dataset):
#     y_true = dataset.get_label()
#     next_close_real = X_train.next_close_real#dataset.get_data()['next_close_real']
#     
#     # 计算奖励
#     reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
#     reward_mean =  np.mean(reward)
#     loss = -reward_mean
#     
#     # print(reward_mean)
#     # reward_all = np.prod(reward + 1)  # 加1以避免负值的影响
# 
#     # 定义参数 alpha 和 beta
#     beta = 0.966 # 控制非对称性, 提高可以促成更多成交 [0.5, 1]
# 
#     # 我们希望最大化奖励，所以我们最小化负奖励
#     
#     # 计算梯度
#     grad = np.where(y_pred <= y_true, 
#                  y_pred - y_true,    # 当 y_pred <= y_true 时，应该增加 y_pred
#                 (y_pred - y_true)*2)     # 当 y_pred > y_true 时，梯度为0因为此时使用 next_close_real
#  
#     # 计算海森矩阵
#     # 计算二阶导数
#     #hess = np.ones_like(y_pred)
#     hess = np.where(y_pred <= y_true,
#                     1* (1 - beta),
#                     0.1* beta)  # 给 `y_pred > y_true` 时更小的步长
#     return grad, hess
# 
# =============================================================================
# 自定义评估指标函数
def custom_metric(y_pred, dataset):
    y_true = dataset.get_label()
    
    #global dataset1#.params
    data_type=dataset.params.get('data_type', '')
    #print(data_type)
   # data = dataset.get_data()
    #next_close_real_index = dataset.feature_name.index('next_close_real')
    #next_close_real = data[:, next_close_real_index]
    #next_close_real = dataset.get_data()['next_close_real']
    if data_type == 'train':
        # 处理训练集的逻辑
        next_close_real = X_train.next_close_real
    elif data_type == 'valid':
        # 处理验证集的逻辑
        next_close_real = X_test.next_close_real
    
    reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
    #reward = np.where(y_pred <= y_true, y_pred, next_close_real)
    #total_profit = np.sum(reward)
    #total_profit = np.prod(reward, axis=0)**0.5  # result.y_pred.prod()
    total_profit = np.mean(reward) 
    
    
    # 计算其他指标
    valid_preds = np.sum(y_pred <= y_true)
    total_preds = len(y_pred)
    valid_ratio = valid_preds / total_preds
    #print(f"Valid predictions ratio: {valid_ratio:.4f} ({valid_preds}/{total_preds})")

    return 'custom_profit', total_profit, True  # 名称, 值, 是否越大越好

#valid_custom_profit:0.00381579
#rmse: 0.03992686340357715
# LightGBM 参数
params = {
    'objective': custom_loss,#'regression',#
    'metric': 'custom', # rmse,# 使用自定义损失函数时，需要设置metric为['None','custom']
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
data = pd.read_csv(f'{PATH}/data/test_603893.csv')
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
# 计算 MSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"rmse: {rmse}")

result = pd.DataFrame([y_test.values, y_pred]).T
result.columns = ['y_test','y_pred']
print(result)
result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

result.to_csv(f'{PATH}/data/test_result_reward2.csv',index=False)
# 训练模型，使用自定义损失函数
