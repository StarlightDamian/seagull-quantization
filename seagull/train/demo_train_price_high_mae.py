# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:10:32 2024

@author: awei
最高价卖出价格模型示例(demo_train_price_high_mae)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from seagull.settings import PATH
# 定义自定义 MAE 损失
# =============================================================================
# def custom_mae_loss(y_pred, dataset):
#     y_true = dataset.get_label()
#     delta = y_true - y_pred
# 
#     # 梯度和 Hessian
#     grad = -np.sign(delta)  # MAE 的梯度为 -sign(delta)
#     hess = np.ones_like(delta) * 1e-9  # 设置 Hessian 为一个小的常数，防止除零错误
# 
#     return grad, hess
# =============================================================================
def custom_mae_loss(y_pred, dataset):
    y_true = dataset.get_label()
    delta = y_true - y_pred + 0.01  # 添加小偏差避免梯度消失
    grad = -np.sign(delta)  # 梯度仍然是符号函数
    hess = np.ones_like(delta) * 0.01  # 将 Hessian 设置为小常数
    return grad, hess

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


# 定义参数
params = {
    'objective': custom_mae_loss,  # 使用自定义 MAE 损失
    'metric': 'None',  # 使用自定义损失时，设置 metric 为 None
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'max_depth': -1,
}

# 训练模型
model = lgb.train(
    params=params,
    train_set=lgb_train_1,
    valid_sets=[lgb_train_1, lgb_valid_1],
    num_boost_round=200
)

# 预测
y_pred = model.predict(X_test_1)

result = pd.DataFrame([y_test.values, y_pred]).T
result.columns = ['y_test','y_pred']
print(result)
result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

result.to_csv(f'{PATH}/_file/test_result_reward2.csv',index=False)

result_bool = result[result.next_high_bool==1]
y_test,y_pred,next_high_bool = result_bool.mean()
print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])

reward = (y_pred-1)*(result_bool.shape[0])
reward_all = (y_test-1)*(result.shape[0])
reward_pct = reward/reward_all
print(f'reward_pct: {reward_pct:.4f}')