# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:02:44 2023

@author: awei
strategy_lightgbm
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设你有一个包含特征和目标值的数据框 df
# 这里的特征和目标值需要根据你的实际情况调整

# 假设特征为 X，目标值为 y

df = pd.DataFrame([['2023-01-03',4.4200], ['2023-01-04', 10.7100],['2023-01-05',5.3800],['2023-01-06',10.6900]], columns=['Date', 'Close'])
X = df.drop(['Date', 'Close'], axis=1)  # 假设特征中不包括日期和收盘价
y = df['Close']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])# , early_stopping_rounds=10

# 预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 可以使用训练好的模型进行未来股票价格的预测
#future_data = ...  # 未来的特征数据
#future_pred = bst.predict(future_data, num_iteration=bst.best_iteration)
