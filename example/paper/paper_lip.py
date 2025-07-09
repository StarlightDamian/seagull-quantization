# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 18:35
@Author: Damian
@Email: zengyuwei1995@163.com
@File: paper_lip.py
@Description: Why Can Accurate Models Be Learned from Inaccurate Annotations?
https://github.com/Chongjie-Si/LIP?tab=readme-ov-file
https://arxiv.org/pdf/2505.16159
"""
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def lip(W_noisy: torch.Tensor,
              X: torch.Tensor,
              Y: torch.Tensor,
              k: int) -> torch.Tensor:
    """
    Apply the LIP refinement to a weight matrix learned from noisy labels,
    using PyTorch tensors.

    Args:
        W_noisy (torch.Tensor): shape (q, l), noisy weight matrix.
        X       (torch.Tensor): shape (n, q), feature matrix.
        Y       (torch.Tensor): shape (n, l), label matrix.
        k       (int): number of principal singular components to retain.

    Returns:
        torch.Tensor: shape (q, l), refined weight matrix W*.
    """
    # 1. Full SVD decomposition on the same device as W_noisy
    U, S, Vh = torch.linalg.svd(W_noisy, full_matrices=False)
    V = Vh.transpose(-2, -1)  # shape (l, l)

    # 2. Principal Subspace Preservation (PSP)
    U_k = U[:, :k]               # (q, k)
    S_k = torch.diag(S[:k])      # (k, k)
    V_k = V[:, :k]               # (l, k)
    W_k = U_k @ S_k @ V_k.t()    # (q, l)

    # 3. Residual subspace
    U_l = U[:, k:]               # (q, r)
    V_l = V[:, k:]               # (l, r)

    # 4. Label Ambiguity Purification (LAP)
    residual = Y - X @ W_k       # (n, l)
    A = U_l.t() @ (X.t() @ residual) @ V_l  # (r, r)
    B = U_l.t() @ (X.t() @ (X @ U_l))       # (r, r)

    # Extract diagonal entries and compute refined singulars
    diag_idx = torch.arange(A.shape[0], device=A.device)
    sigma_refined = A[diag_idx, diag_idx] / B[diag_idx, diag_idx]
    S_l_refined = torch.diag(sigma_refined)  # (r, r)

    # 5. Reconstruct final weight
    W_refined = W_k + U_l @ S_l_refined @ V_l.t()  # (q, l)
    return W_refined


# 创建回归数据集
X, y = make_regression(
    n_samples=1000,  # 样本数
    n_features=20,   # 特征数
    n_informative=15, # 有效特征数
    noise=0.5,       # 噪声水平
    random_state=42
)

# 添加标签噪声（模拟标签错误）
np.random.seed(42)
noise_mask = np.random.choice([0, 1], size=y.shape, p=[0.8, 0.2])
y_noisy = y + noise_mask * np.random.normal(0, 50, size=y.shape)  # 20%的标签添加大噪声

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
y_noisy_tensor = torch.tensor(y_noisy, dtype=torch.float32).unsqueeze(1)

# 1. 在干净标签上训练的模型（理想情况）
clean_model = LinearRegression()
clean_model.fit(X, y)
clean_pred = clean_model.predict(X)
clean_mse = mean_squared_error(y, clean_pred)

# 2. 在噪声标签上训练的模型（现实情况）
noisy_model = LinearRegression()
noisy_model.fit(X, y_noisy)
noisy_pred = noisy_model.predict(X)
noisy_mse = mean_squared_error(y, noisy_pred)

# 3. 应用LIP修正后的模型
# 提取噪声模型的权重
W_noisy = torch.tensor(noisy_model.coef_, dtype=torch.float32).unsqueeze(1)

# 应用LIP修正
W_refined = lip(
    W_noisy=W_noisy,
    X=X_tensor,
    Y=y_tensor,
    k=10  # 保留10个主成分
)

# 创建修正后的模型
refined_model = LinearRegression()
refined_model.coef_ = W_refined.squeeze().detach().numpy()
refined_model.intercept_ = 0  # 假设无截距项
refined_pred = refined_model.predict(X)
refined_mse = mean_squared_error(y, refined_pred)

# 打印结果
print(f"干净标签模型MSE: {clean_mse:.4f}")
print(f"噪声标签模型MSE: {noisy_mse:.4f} (性能下降 {((noisy_mse-clean_mse)/clean_mse)*100:.1f}%)")
print(f"LIP修正后模型MSE: {refined_mse:.4f} (提升 {((noisy_mse-refined_mse)/noisy_mse)*100:.1f}%)")