# -*- coding: utf-8 -*-
"""
@Date: 2025/6/25 14:33
@Author: Damian
@Email: zengyuwei1995@163.com
@File: 矩阵2.py
@Description:
"""
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.linalg import block_diag

# -----------------------
# 1. 参数与模拟数据（同原来）
# -----------------------
T, N, Q = 4, 2, 3
np.random.seed(0)
P_t = np.random.uniform(50, 150, size=(T, N))
quantile_base = np.linspace(0.9, 1.1, Q)
price_3d = P_t[:, :, None] * (
    quantile_base[None, None, :]
    + 0.1 * np.random.randn(T, N, Q)
)  # shape (T, N, Q)

# -----------------------
# 2. 一次性计算 mu_vec (T*N,) 和 Sigmas_list
# -----------------------

# 2.1 简单收益 R and mu_t
# R: shape (T, N, Q) of simple returns
R = price_3d / P_t[:, :, None] - 1
mu = R.mean(axis=2)              # shape (T, N)
mu_vec = mu.ravel(order='C')     # flatten to (T*N,)

# 2.2 对数收益 L and 协方差 Sigma_t
L = np.log(price_3d / P_t[:, :, None])  # shape (T, N, Q)
# 去中心化
Lc = L - L.mean(axis=2, keepdims=True)  # same shape
# 计算每个 t 的协方差：Σ_t = Lc_t @ Lc_t^T / (Q-1)
# 利用 Einstein 求和一次性得到 (T, N, N)
Sigma_t = np.einsum('tnq,tmq->tnm', Lc, Lc) / (Q-1)  # shape (T, N, N)

# 2.3 把所有 Σ_t 堆成大块对角矩阵 Σ_big
#    用 SciPy.block_diag 对 *args
Sigma_big = block_diag(*Sigma_t)

# -----------------------
# 3. 定义并解 QP —— 与原来一致
# -----------------------
w = cp.Variable(T*N)
objective = cp.Maximize(mu_vec @ w - 0.5 * cp.quad_form(w, Sigma_big))
constraints = [
    *[
        cp.sum(w[t*N:(t+1)*N]) == 1
        for t in range(T)
    ],
    w >= 0
]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)

W_opt = w.value.reshape(T, N)

# -----------------------
# 4. 展示结果
# -----------------------
df_weights = pd.DataFrame(
    W_opt,
    index=[f"Day_{t+1}" for t in range(T)],
    columns=[f"Stock_{i+1}" for i in range(N)]
)
df_mus = pd.DataFrame(
    mu,
    index=[f"Day_{t+1}" for t in range(T)],
    columns=[f"Stock_{i+1}" for i in range(N)]
)

print("mu_t (simple returns):")
print(df_mus.round(4))
print("\nOptimal weights:")
print(df_weights.round(4))
