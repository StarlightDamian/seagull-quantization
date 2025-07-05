# -*- coding: utf-8 -*-
"""
@Date: 2025/7/4 23:09
@Author: Damian
@Email: zengyuwei1995@163.com
@File: paper_Why Naive 1N Diversification Is Not So Naive and How to Beat It.py
@Description:
Why Naive 1N Diversification Is Not So Naive and How to Beat It
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3991279

简单来说，这篇文章告诉我们三件事：

1. **“把钱平均分”其实没那么笨**
   当你手头的资产很多、可用的数据又不算特别多时，把钱平均分配到每个资产（1/N）往往比那些看起来更“聪明”的基于历史估计的配比方法跑得好——因为估计误差太大，反而拖累了收益。

2. **数据够多时，可以“参照估计，打个折再用”**
   如果你的资产数量比数据时期数少（比如你有 10 只股票，但有 100 个月的历史数据），那就可以把“平均分配”跟“历史估计出来的最优配比”用一个最优比例调和起来，这样既兼顾了平均分配的稳定，又加入了估计配比的优势，能跑得更好。

3. **当资产比数据更多，还得靠“选出有用信号”**
   如果你想配置的资产比你能用的数据还多（比如你有上千只股票但只有几百个月的数据），光用历史估计根本不够，这时候就得引入额外的预测信号（像各种因子或者机器学习出来的“alpha”）来帮忙，才能超过简单的平均分配。

总结：
* 数据少、资产多时，1/N 最靠谱；
* 数据多一些时，1/N＋估计配比的混合更好；
* 资产太多、数据有限时，要靠选信号才能打败 1/N。

"""
import numpy as np

# 假设：已知资产收益矩阵 returns (T×N)，无风险率 rf
T, N = returns.shape
excess = returns - rf  # T×N

# 1. Plug-in 最优权重
mu_hat   = excess.mean(axis=0)              # N
Sigma_hat= np.cov(excess, rowvar=False)     # N×N
w_pi     = np.linalg.solve(Sigma_hat, mu_hat)
w_pi    /= np.sum(w_pi)                     # 可选归一化

# 2. GMV 权重
ones     = np.ones(N)
w_gmv    = np.linalg.solve(Sigma_hat, ones)
w_gmv   /= ones @ w_gmv

# 3. Three‑fund 组合 (plug‑in 与 GMV 线性组合)
lambda_ = 0.5  # 任意选择，可用解析式优化
w_3fund = (1-lambda_)*w_pi + lambda_*w_gmv

# 4. Naive 1/N
w_1n     = np.ones(N) / N

# 5. 当 N<T 时，对 1/N 与 plug‑in 组合并求最优 λ
#    亦可用闭式 λ*，此处示意数值优化
import scipy.optimize as opt

def sharpe_comb(lambda_):
    w = lambda_*w_pi + (1-lambda_)*w_1n
    port_ret = excess @ w
    return - port_ret.mean() / port_ret.std()

res = opt.minimize_scalar(sharpe_comb, bounds=(0,1), method='bounded')
lambda_star = res.x
w_optimal   = lambda_star*w_pi + (1-lambda_star)*w_1n

# 6. 当 N>T 时，加入 alpha 信号组合示例
#    alphas: 长短组合月度收益序列 (T,)
alpha_signal = anomaly_returns  # 例如 top5% anomalies 组合
X             = np.column_stack([alpha_signal, excess @ w_1n])
# 线性组合系数 = Σ^{-1} μ
mu2           = X.mean(axis=0)          # 长短策略 & 1/N 组合
Sigma2        = np.cov(X, rowvar=False)
w2            = np.linalg.solve(Sigma2, mu2)
w2           /= np.sum(np.abs(w2))      # L1 归一化以控制杠杆

# 最终权重：w2[0]*alpha + w2[1]*1/N
