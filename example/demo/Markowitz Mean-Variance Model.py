# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 18:39
@Author: Damian
@Email: zengyuwei1995@163.com
@File: Markowitz Mean-Variance Model.py
@Description: 
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:07:33 2025

@author: Damian
"""

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier

# —— 1. 构造示例输入 ——
tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]


np.random.seed(42)
mu = pd.Series(
    np.random.uniform(0.05, 0.15, len(tickers)),
    index=tickers,
    name="Expected Return"
)

# 随机生成一个对称正定的协方差矩阵 Σ
M = np.random.randn(len(tickers), len(tickers))
Sigma = pd.DataFrame(
    M.T.dot(M) * 0.1,
    index=tickers,
    columns=tickers
)

# 输入展示
print("=== Input: Expected Returns (μ) ===")
print(mu.to_frame())
print("\n=== Input: Covariance Matrix (Σ) ===")
print(Sigma)

# —— 2. 优化 ——
ef = EfficientFrontier(mu, Sigma)
ef.max_sharpe()
weights = ef.clean_weights()      # 取非零且格式化后的权重字典

# 转成 DataFrame
weights_df = pd.Series(weights, name="Weight").to_frame()
print("\n=== Output: Optimized Weights ===")
print(weights_df)
