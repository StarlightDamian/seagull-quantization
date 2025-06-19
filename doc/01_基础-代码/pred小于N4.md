# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:09:29 2024

@author: awei
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# 自定义目标函数
def custom_asymmetric_objective(y_pred, dataset):
    y_true = dataset.get_label()  # Extract actual values
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * np.abs(residual), 0.1 * np.abs(residual))  # Increase gradient for under-prediction
    hess = np.ones_like(residual)
    return grad, hess


def custom_asymmetric_eval(y_pred, dataset):
    y_true = dataset.get_label()
    residual = y_true - y_pred
    loss = np.where(residual > 0, 
                    (residual ** 2) * 2.0,  # Stronger penalty for under-prediction
                    (residual ** 2) * 0.1)  # Weaker penalty for over-prediction
    return "custom_asymmetric_eval", np.mean(loss), False


# Generate random data to simulate stock prices
np.random.seed(42)
X = pd.DataFrame(np.random.randn(10000, 10), columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(np.random.uniform(50, 150, 10000), name='stock_price')  # Simulated stock prices


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': custom_asymmetric_objective,
    'metric': 'custom',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 训练模型
model = lgb.train(params, train_data, num_boost_round=100, 
                  valid_sets=[test_data], 
                  feval=custom_asymmetric_eval,
                  #early_stopping_rounds=10
                  )

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 计算预测值低于实际值的比例
below_actual = np.mean(y_pred < y_test)
print(f"Proportion of predictions below actual: {below_actual:.2%}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

# 分析预测误差
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# 自定义评估函数
def custom_metric(y_true, y_pred):
    below_actual = np.mean(y_pred < y_true)
    mse = np.mean((y_true - y_pred)**2)
    return below_actual, mse

below_actual, mse = custom_metric(y_test, y_pred)
print(f"Custom Metric - Below Actual: {below_actual:.2%}, MSE: {mse:.4f}")