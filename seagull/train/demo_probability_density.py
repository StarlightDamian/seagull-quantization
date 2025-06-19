# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:16:48 2024

@author: awei
"""
# =============================================================================
# from sklearn.neighbors import KernelDensity
# kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data)
# log_density = kde.score_samples(X)  # X 是待预测数据
# density = np.exp(log_density)  # 转换为概率密度
# =============================================================================
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# Custom objective function
def custom_asymmetric_objective(y_pred, dataset):
    y_true = dataset.get_label()  # Extract actual values
    residual = y_true - y_pred
    grad = np.where(residual > 0, -0.86 * np.abs(residual), 0.14 * np.abs(residual))  # Larger gradient for bigger errors
    hess = np.ones_like(residual)  # Hessian
    return grad, hess


# Custom evaluation function
def custom_asymmetric_eval(y_pred, dataset):
    y_true = dataset.get_label()  # Extract actual values
    residual = y_true - y_pred
    loss = np.where(residual > 0, 
                    (residual ** 2) * 0.86,  # Heavier penalty when under-predicting
                    (residual ** 2) * 0.14)  # Lighter penalty when over-predicting
    return "custom_asymmetric_eval", np.mean(loss), False


np.random.seed(43)
X = pd.DataFrame(np.random.randn(10000, 10), columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(np.random.uniform(50, 150, 10000), name='num') 
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set parameters
params = {
    'boosting_type': 'gbdt',
    'objective': custom_asymmetric_objective,  # Custom objective
    'metric': 'custom',  # Custom metric
    'num_leaves': 100,  # Increase number of leaves to capture more patterns
    'learning_rate': 0.05,
    'max_depth': 10,  # Add a max depth to prevent overfitting
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


# Train the model
model = lgb.train(params, train_data, num_boost_round=100,
                  valid_sets=[test_data],
                  feval=custom_asymmetric_eval,
                  #early_stopping_rounds=10
                  )

# Predict
y_pred = model.predict(X_test)


# Fit KDE on the residuals (errors)
errors = y_test - y_pred
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(errors.values.reshape(-1, 1))

# 生成用于评估密度
X_plot = np.linspace(errors.min() - 1, errors.max() + 1, 1000).reshape(-1, 1)

# 计算对数密度
log_density = kde.score_samples(X_plot)
density = np.exp(log_density)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(X_plot, density, label="Density of Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Probability Density")
plt.title("Probability Density of Prediction Errors using KDE")
plt.legend()
plt.tight_layout()
plt.show()
