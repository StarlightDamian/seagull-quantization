# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 11:10
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_lgb_volatility.py
@Description: ligbm 回报率预测的自定义损失函数示例  应该预测两端极值
"""
# -*- coding: utf-8 -*-

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# —— 1. 造个回报率预测数据示例 ——
X, y = make_regression(n_samples=2000, n_features=10, noise=0.1, random_state=42)
# 把 y 缩放成“回报率”范围，大约 ±10%
y = y / np.max(np.abs(y)) * 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


# —— 2. 自定义损失：Thresholded Absolute Loss ——
def threshold_loss(preds, dataset):
    y_true = dataset.get_label()
    e = preds - y_true
    eps = 0.02  # 2% 阈值
    # 只有 |e|>=eps 才计入损失
    mask = np.abs(e) >= eps

    # 损失 L = |e| for |e|>=eps, else 0
    grad = np.zeros_like(e)
    hess = np.zeros_like(e)

    # 梯度：dL/de = sign(e) for mask
    grad[mask] = np.sign(e[mask])
    # 黑塞：二阶导 ∂²L/∂e² = 0 (对绝对值而言除非 e=0)
    # 为了数值稳定，我们给一个极小正值
    hess[mask] = 1e-6

    return grad, hess


# —— 3. 训练 LightGBM with custom objective ——
params = {
    'objective': threshold_loss,  # 'regression',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'metric': 'None'  # 评估时我们单独再算定制损失
}
bst = lgb.train(
    params,  # dict 中不要再放 'objective'，它用默认的回归 loss
    train_data,
    num_boost_round=200,  # 放到第三个位置或用关键字
    # objective=threshold_loss,  # 自定义目标
    feval=lambda preds, ds: (
        'thresh_loss',
        np.mean(np.abs(preds - ds.get_label()) * (np.abs(preds - ds.get_label()) >= 0.02)),
        False
    ),
    valid_sets=[valid_data],
    # early_stopping_rounds=20,
)
# =============================================================================
# bst = lgb.train(
#     params,
#     train_data,
#     feval=lambda preds, ds: (
#         'threshold_loss',
#         np.mean(np.abs(preds - ds.get_label()) * (np.abs(preds - ds.get_label())>=0.02)),
#         False
#     ),
#     fobj=threshold_loss,
#     valid_sets=[valid_data],
#     num_boost_round=200,
#     early_stopping_rounds=20
# )
#
# =============================================================================
# —— 4. 模型评估 ——
preds = bst.predict(X_val)
errors = preds - y_val
mask = np.abs(errors) >= 0.02
print("Avg. Thresh-Loss:", np.mean(np.abs(errors[mask])) if mask.any() else 0.0)
