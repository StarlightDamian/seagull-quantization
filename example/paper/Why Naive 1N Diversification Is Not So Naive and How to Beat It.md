简单来说，这篇文章告诉我们三件事：

1. **“把钱平均分”其实没那么笨**
    当你手头的资产很多、可用的数据又不算特别多时，把钱平均分配到每个资产（1/N）往往比那些看起来更“聪明”的基于历史估计的配比方法跑得好——因为估计误差太大，反而拖累了收益。
2. **数据够多时，可以“参照估计，打个折再用”**
    如果你的资产数量比数据时期数少（比如你有 10 只股票，但有 100 个月的历史数据），那就可以把“平均分配”跟“历史估计出来的最优配比”用一个最优比例调和起来，这样既兼顾了平均分配的稳定，又加入了估计配比的优势，能跑得更好。
3. **当资产比数据更多，还得靠“选出有用信号”**
    如果你想配置的资产比你能用的数据还多（比如你有上千只股票但只有几百个月的数据），光用历史估计根本不够，这时候就得引入额外的预测信号（像各种因子或者机器学习出来的“alpha”）来帮忙，才能超过简单的平均分配。

总的来说，就是：

* 数据少、资产多时，1/N 最靠谱；
* 数据多一些时，1/N＋估计配比的混合更好；
* 资产太多、数据有限时，要靠选信号才能打败 1/N。

文章《Why Naive 1/N Diversification Is Not So Naive and How to Beat It》的三大核心贡献如下：

1. **高维情形下常见估计规则的Sharpe比率分析**

   * 推导了Plug‑in、GMV（Global Minimum Variance）和三基金（three‑fund）规则在 $N/T\to\eta\in(0,1)$ 时的渐近Sharpe比率 $\tau$，并表明只要资产数 $N$ 与样本期数 $T$ 比例不趋零，这些估计规则都无法收敛到无误差下的最优Sharpe比率（Proposition 2）。

   * 给出Plug‑in规则 Sharpe 比率的渐近偏差因子

     SRplug‑in≈τPI×SR,τPI=1−η1+η/SR2<1.  \mathrm{SR}_{\text{plug‑in}}  \approx \tau_{\text{PI}}\times\mathrm{SR},\quad  \tau_{\text{PI}}=\sqrt{\frac{1-\eta}{1+\eta/\mathrm{SR}^2}}<1.

   * 给出GMV规则渐近因子

     SRGMV≈τGMV×SR,τGMV=ρ1−η<1,  \mathrm{SR}_{\text{GMV}}  \approx \tau_{\text{GMV}}\times\mathrm{SR},\quad  \tau_{\text{GMV}}=\rho\sqrt{1-\eta}<1,

     其中 $\rho=\mathrm{SR}_\mathrm{GMV}/\mathrm{SR}$。

2. **在一因子模型下 $1/N$ 最优性**

   * 若资产超额收益 $R_i = \beta_i R_q + \varepsilon_i$，且 idiosyncratic 风险足够可分散，$\tfrac1N\sum\beta_i\to\beta_0>0$，那么等权组合的Sharpe比率

     SR1/N  =  SR+O(N−1/2),  \mathrm{SR}_{1/N} \;=\;\mathrm{SR}+O\bigl(N^{-1/2}\bigr),

     当 $N\to\infty$ 时趋近最优（Proposition 3）。这解释了为何在高维资产池中，1/N rule 往往难以被估计规则超越。

3. **有条件击败 $1/N$ 的可行策略**

   * **当 $N 时**，分别将 $1/N$ 与 Plug‑in 或 GMV 规则线性组合：

     wλ=λ west+(1−λ) 1N1N,λ∗=arg⁡max⁡τ(λ).  w_{\lambda} = \lambda\,w_{\text{est}} + (1-\lambda)\,\tfrac1N\mathbf1_N,\quad  \lambda^*=\arg\max\tau(\lambda).

     明确给出最优 $\lambda^*$ 的解析形式（Proposition 5、7），并在实证中验证小维度时组合优于单一1/N。

   * **当 $N>T$ 时**，由于协方差不可逆，可借助“alpha”信息：

     * 将1/N与基于 **anomalies**（选取预测alpha排名前 $\omega\%$ 的策略）或 **机器学习长短组合**（如GBR、随机森林、神经网络）组合，显著提升Sharpe，条件于这些信号的有效性。

------

### 用 Python 公式复现核心组合策略

```python
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
```

以上代码示意了论文中各类组合策略的关键公式与实现思路。