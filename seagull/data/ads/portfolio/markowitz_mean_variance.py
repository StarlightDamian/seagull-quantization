# -*- coding: utf-8 -*-
"""
@Date: 2025/6/22 19:43
@Author: Damian
@Email: zengyuwei1995@163.com
@File: markowitz_mean_variance.py
@Description:  风险 - 收益权衡优化
使用 CVXPY 库求解了一个 投资组合优化问题，具体来说是 马克维茨均值 - 方差模型 的变体
最大化：预期收益 - λ × 风险
约束：权重总和为1，所有权重非负（不允许卖空）

预期收益：m.T @ w（资产预期收益率的加权和）
风险：0.5*cp.quad_form(w, Sigma)（投资组合方差，即风险）
λ：风险厌恶系数（这里隐含在 Sigma 中）
约束条件：
    1. A @ w == 1：权重总和为 1（即资金全部投入）
    2. w >= 0：不允许卖空（所有权重非负）
"""
import cvxpy as cp

# 创建优化变量：w是一个长度为5*T的向量
# 这里假设：有5种资产，T个时间周期，w表示每个资产在每个周期的权重
w = cp.Variable(5*T)

# 定义目标函数：最大化 预期收益 - 风险
# m：预期收益率向量（长度5*T）
# Sigma：协方差矩阵（形状为(5*T, 5*T)）
obj = cp.Maximize(m.T @ w - 0.5*cp.quad_form(w, Sigma))

# 定义约束条件
# A @ w == 1：权重总和为1（投资完全满仓）
# w >= 0：所有权重非负（不允许卖空）
constraints = [A @ w == 1, w >= 0]

# 构建并求解优化问题
# 使用OSQP求解器（适用于二次规划问题）
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.OSQP)

# 将结果重塑为(5, T)矩阵
# 即每种资产在每个时间周期的最优权重
w_opt = w.value.reshape(5, T)
