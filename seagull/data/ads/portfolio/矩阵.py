# -*- coding: utf-8 -*-
"""
@Date: 2025/6/22 19:37
@Author: Damian
@Email: zengyuwei1995@163.com
@File: 矩阵.py
@Description: 
"""
import numpy as np
import pandas as pd
import cvxpy as cp

# -----------------------
# 1. Simulate a 3D matrix of predicted quantile prices
# -----------------------
# Dimensions: T days, N stocks, Q quantiles
T, N, Q = 4, 2, 3
np.random.seed(0)

# Simulate current prices P_t (for simplicity, same for all stocks each day)
P_t = np.random.uniform(50, 150, size=(T, N))

# Simulate predicted quantile future prices: shape (T, N, Q)
# Ensure increasing across quantiles for realism
quantile_base = np.linspace(0.9, 1.1, Q)
price_3d = np.array([
    P_t[t][:, None] * (quantile_base + 0.1 * np.random.randn(N, Q))
    for t in range(T)
])

# -----------------------
# 2. Compute for each day t:
#    - mu_t: expected simple returns per stock (weighted by uniform prob 1/Q)
#    - Sigma_t: covariance of log returns per stock
# -----------------------
mus = []      # list of length T, each is (N,) vector
Sigmas = []   # list of length T, each is (N,N) matrix

for t in range(T):
    prices = price_3d[t]          # shape N x Q
    P0 = P_t[t]                   # current prices, shape (N,)
    # simple returns for each quantile: R[i,q] = p_iq / P0_i - 1
    R = prices / P0[:, None] - 1  # shape N x Q
    # expected simple return mu_t
    mu_t = R.mean(axis=1)         # shape (N,)
    mus.append(mu_t)
    # log returns for covariance
    L = np.log(prices / P0[:, None])  # N x Q
    # weighted covariance (Q samples, equiprobable)
    L_centered = L - L.mean(axis=1, keepdims=True)
    Sigma_t = (L_centered @ L_centered.T) / (Q - 1)
    Sigmas.append(Sigma_t)

# Stack into big vectors/matrices for multi-period optimization
# mu_vec shape (N*T,)
mu_vec = np.concatenate(mus, axis=0)

# Build block-diagonal Sigma of shape (N*T, N*T)
Sigma_big = np.zeros((N*T, N*T))
for t in range(T):
    Sigma_big[t*N:(t+1)*N, t*N:(t+1)*N] = Sigmas[t]

# -----------------------
# 3. Define and solve the QP:
#    max_w mu^T w - 0.5 w^T Sigma w
#    s.t. for each t: sum_{i=1}^N w_{t,i} = 1, w>=0
# -----------------------
w = cp.Variable(N*T)

# Objective
objective = cp.Maximize(mu_vec @ w - 0.5 * cp.quad_form(w, Sigma_big))

# Constraints: sum of weights per day = 1, and non-negative
constraints = []
for t in range(T):
    constraints.append(cp.sum(w[t*N:(t+1)*N]) == 1)
constraints.append(w >= 0)

# Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)

# Reshape weights to matrix W (T x N)
W_opt = w.value.reshape(T, N)

# -----------------------
# 4. Display results
# -----------------------
df_weights = pd.DataFrame(
    W_opt,
    index=[f"Day_{t+1}" for t in range(T)],
    columns=[f"Stock_{i+1}" for i in range(N)]
)

df_mus = pd.DataFrame(
    np.stack(mus),
    index=[f"Day_{t+1}" for t in range(T)],
    columns=[f"Stock_{i+1}" for i in range(N)]
)

print("Expected Simple Returns (mu_t) per day and stock:")
print(df_mus.round(4))
print("\nOptimal Weights per day and stock:")
print(df_weights.round(4))
