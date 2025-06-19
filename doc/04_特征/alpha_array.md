下面针对你提供的这些常用滚动与变换函数，一一说明它们的功能、可优化之处，并在代码中加上详细注释。总体思路是：

1. 利用 Pandas 原生函数（`.rolling()`, `.rank()` 等）实现“时间序列”与“横截面”逻辑
2. 尽量避免冗余代码（例如重复定义、手写循环）
3. 注意去除会修改输入 `DataFrame` 的 `inplace=True`（除非确实需要就地修改），以免带来意料之外的副作用
4. 对于某些自定义的滚动聚合，可考虑用 `numba`、`cython`、或 `pandas` 内部优化方法进一步加速（此处以 Pandas 原生为主）。

------

```python
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# --------------------------------------------------------------------------------
# 0. Rolling Sum（滚动求和）
#    等价于：df.rolling(window).sum()
# --------------------------------------------------------------------------------
def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算过去 window 个交易日的累积和。
    :param df: pandas DataFrame，行索引为日期，列索引为股票/品种等。
    :param window: 滚动窗口大小（天数）。
    :return: 每个日期的“过去 window 天”窗口内求和结果。
    """
    # 直接调用 pandas 的 rolling.sum 即可，高效且清晰
    return df.rolling(window).sum()

# --------------------------------------------------------------------------------
# 1. Simple Moving Average（简单移动平均）
#    等价于：df.rolling(window).mean()
# --------------------------------------------------------------------------------
def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算简单移动平均（SMA）。
    :param df: pandas DataFrame，行情数据等。
    :param window: 滚动窗口大小。
    :return: 每个日期的“过去 window 天”平均值。
    """
    return df.rolling(window).mean()

# --------------------------------------------------------------------------------
# 2. Rolling Standard Deviation（滚动标准差）
#    等价于：df.rolling(window).std()
# --------------------------------------------------------------------------------
def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算滚动标准差（波动率）。
    :param df: pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的标准差。
    """
    return df.rolling(window).std()

# --------------------------------------------------------------------------------
# 3. Rolling Correlation（滚动相关系数）
#    等价于：x.rolling(window).corr(y)
# --------------------------------------------------------------------------------
def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算 x 与 y 在过去 window 天的逐日滚动皮尔逊相关系数。
    :param x: pandas DataFrame 或 Series，行情/因子序列。
    :param y: pandas DataFrame 或 Series，与 x 形状需对应（同索引）。
    :param window: 滚动窗口大小。
    :return: DataFrame 或 Series，表示每个日期的滚动相关系数。
    """
    return x.rolling(window).corr(y)

# --------------------------------------------------------------------------------
# 4. Rolling Covariance（滚动协方差）
#    等价于：x.rolling(window).cov(y)
# --------------------------------------------------------------------------------
def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算 x 与 y 在过去 window 天的滚动协方差。
    :param x: pandas DataFrame 或 Series。
    :param y: pandas DataFrame 或 Series，与 x 索引对齐。
    :param window: 滚动窗口大小。
    :return: 每个日期的滚动协方差。
    """
    return x.rolling(window).cov(y)

# --------------------------------------------------------------------------------
# 5. Rolling Rank（滚动排序/分位）
#    先定义辅助函数 rolling_rank：返回窗口内最后一个元素在该窗口内的排名（百分比）
# --------------------------------------------------------------------------------
def rolling_rank(arr: np.ndarray) -> float:
    """
    帮助函数：返回 arr（长度为 window）中最后一个元素的横截面排序位置（百分比）。
    :param arr: numpy 数组，长度等于窗口大小。
    :return: float，表示最后一个元素在 arr 中的 rank pct。
    """
    # rankdata 默认给出 1 到 n 的排名，这里以百分比形式返回
    ranks = rankdata(arr)       # 生成 [1, 2, ..., window] 排名
    return ranks[-1] / len(arr)  # 将 rank 转换为 [1/window, 2/window, ...]，最后一个位置的值

def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算每个日期对应值在过去 window 天内的排名百分比。
    :param df: pandas DataFrame，行索引为日期。
    :param window: 滚动窗口大小。
    :return: DataFrame，与 df 同形状，每个元素表示该列在过去 window 天的 ts_rank。
    """
    # 直接调用 rolling.apply 并传入自定义函数
    # 注意：rolling.apply 默认 axis=0，对每列做窗口，rolling_rank 会得到最后一个值的排名百分比
    return df.rolling(window).apply(rolling_rank, raw=True)

# --------------------------------------------------------------------------------
# 6. Rolling Product（滚动乘积）
#    等价于：df.rolling(window).apply(lambda arr: np.prod(arr))
#    如果窗口较大，此处可能比较慢，可视情况改用 numba 加速
# --------------------------------------------------------------------------------
def rolling_prod(arr: np.ndarray) -> float:
    """
    辅助函数：返回数组 arr 中所有值的乘积。
    :param arr: numpy 数组。
    :return: float，乘积结果。
    """
    return np.prod(arr)

def product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算过去 window 天每列数据的滚动乘积。
    :param df: pandas DataFrame。
    :param window: 窗口大小。
    :return: 每个日期的过去 window 天的乘积。
    """
    # raw=True：直接传入 numpy 数组给 rolling_prod，更省内存
    return df.rolling(window).apply(rolling_prod, raw=True)

# --------------------------------------------------------------------------------
# 7. Rolling Min & Max（滚动最小/最大值）
#    等价于：df.rolling(window).min()/max()
# --------------------------------------------------------------------------------
def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算过去 window 天的最小值。
    :param df: pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的最小值。
    """
    return df.rolling(window).min()

def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算过去 window 天的最大值。
    :param df: pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: 每个日期过去 window 天的最大值。
    """
    return df.rolling(window).max()

# --------------------------------------------------------------------------------
# 8. Delta（差分）与 Delay（滞后）
# --------------------------------------------------------------------------------
def delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    计算当前值与 period 天前的差：df_t - df_{t-period}。
    :param df: pandas DataFrame。
    :param period: 滞后天数。
    :return: 每列对应的差分结果，首 period 行会产生 NaN。
    """
    return df.diff(period)

def delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    将整个时序向下平移 period 行，用于计算滞后值。
    :param df: pandas DataFrame。
    :param period: 滞后行数。
    :return: df.shift(period)，前 period 行为 NaN。
    """
    return df.shift(period)

# --------------------------------------------------------------------------------
# 9. Cross-Sectional Rank（横截面排序）
#    等价于：df.rank(axis=1, pct=True)
# --------------------------------------------------------------------------------
def rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个日期横截面上，每列在当日所有列中的分位排名（百分比形式）。
    :param df: pandas DataFrame，行索引为日期，列为不同股票/品种。
    :return: DataFrame，与 df 同形状，每列为该值在当日所有列中的排名百分比。
    """
    # axis=1：对每行进行排序，pct=True：返回百分比排名
    return df.rank(axis=1, pct=True)

# --------------------------------------------------------------------------------
# 10. Scale（归一化，使绝对值之和为 k）
# --------------------------------------------------------------------------------
def scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """
    将每列按绝对值之和归一化，使得 sum(abs(df)) = k。常用于因子值归一化。
    :param df: pandas DataFrame。
    :param k: 归一化后总绝对值之和（例如 k=1 表示 L1 归一化）。
    :return: DataFrame，与 df 同形状。
    """
    # abs(df).sum()：按列计算绝对值之和
    # df.div(...)：逐列除以当列的绝对值之和
    return df.mul(k).div(np.abs(df).sum(axis=0))

# --------------------------------------------------------------------------------
# 11. ts_argmax & ts_argmin（滚动最大/最小值所在位置）
#    等价于：df.rolling(window).apply(lambda arr: np.argmax(arr))，再 +1 表示位置
# --------------------------------------------------------------------------------
def ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算每个日期过去 window 天内的最大值出现在距离当前有几天之前。
    :param df: pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: DataFrame，返回值范围为 1 到 window，1 表示当日最大，window 表示 window 天前最大。
    """
    # rolling.apply 会把窗口内数据 (长度=window) 传给 np.argmax
    # np.argmax 返回索引 0..window-1，需要 +1 让它变为 1..window
    return df.rolling(window).apply(lambda arr: np.argmax(arr) + 1, raw=True)

def ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    计算每个日期过去 window 天内的最小值出现在距离当前有几天之前。
    :param df: pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: DataFrame，范围 1..window，1 表示当日最小，window 表示 window 天前最小。
    """
    return df.rolling(window).apply(lambda arr: np.argmin(arr) + 1, raw=True)

# --------------------------------------------------------------------------------
# 12. Linear Decay / Linear Weighted Moving Average（线性加权移动平均）
#    将最近值赋予更高权重，越过去越小
#    代码中出现两次重定义，需要合并
# --------------------------------------------------------------------------------
def decay_linear(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算线性加权移动平均（LWMA），即最近日期赋予更高权重。
    :param df: pandas DataFrame，需要没有 NaN 或先自行填充（以下会做前后填充/补零）。
    :param period: 窗口大小。
    :return: 每个日期过去 period 天的加权平均值，权重为 1,2,...,period / (period*(period+1)/2)。
    """
    # ---- 数据清洗：避免因 NaN 导致计算错误 ----
    if df.isnull().values.any():
        # 前向填充后向填充，再将剩余 NaN 填 0
        df = df.ffill().bfill().fillna(0)
    
    # 将 DataFrame 转成 numpy 数组：shape = (n_dates, n_columns)
    arr = df.to_numpy()
    n_rows, n_cols = arr.shape
    
    # 计算线性权重：最近一天权重 = period / sum(1..period)，最早一天 = 1 / sum(1..period)
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()  # = period*(period+1)/2
    normalized_weights = weights / weight_sum  # 长度 = period
    
    # 初始化结果矩阵
    result = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    
    # 前 period-1 行直接用原始值（可根据需求改为 NaN 或不计算）
    result[:period - 1, :] = arr[:period - 1, :]
    
    # 从第 period-1 行开始做加权平均
    for i in range(period - 1, n_rows):
        window_slice = arr[i - period + 1 : i + 1, :]  # 窗口内数据，shape = (period, n_cols)
        # 每列与 weights 相乘后求和
        # np.dot(window_slice.T, normalized_weights)  等价于 对各列做 dot
        result[i, :] = window_slice.T.dot(normalized_weights)
    
    # 将结果转回 DataFrame，列名与索引沿用原始 df
    return pd.DataFrame(result, index=df.index, columns=df.columns)
```

------

## 逐个函数解析与可优化之处

1. **ts_sum / sma / stddev / ts_min / ts_max**

   - 功能：分别计算“过去 N 天的累积和”“移动平均”“滚动标准差”“滚动最小”“滚动最大”。
   - 优化：直接调用 Pandas 的原生 `.rolling(window).sum()/.mean()/.std()/.min()/.max()` 已足够高效，不需要额外优化。

2. **correlation / covariance**

   - 功能：计算两列（或两组列）之间“过去 N 天”的滚动相关系数或协方差。
   - 优化：直接调用 `x.rolling(window).corr(y)`、`x.rolling(window).cov(y)`，无需自写。
   - 注意：如果要计算成对的多列滚动相关（比如整个因子矩阵与基准矩阵之间的矩阵相关），需要循环或使用更高级的向量化方案。

3. **ts_rank（rolling rank）**

   - 功能：对每一列的时间序列，在窗口内给出“当日值在过去 N 天内的横截面排名百分比”。
   - 实现：由于 Pandas 没有直接提供 rolling 内部排序结果的索引，需要用 `rolling.apply(rolling_rank)`。
   - 优化：
     - `raw=True`：会把窗口数据直接以 `numpy.ndarray` 形式传给 `rolling_rank`，减少额外开销。
     - 如果窗口很大且数据量多，可考虑用专门的排序加速库（如 `numba` 加速）来替代 `scipy.stats.rankdata`，但一般日常 N<=60 或 120 时开销可接受。

4. **product / rolling_prod（滚动乘积）**

   - 功能：计算“过去 N 天所有值乘起来”，常用于“累计回报”之类场景，窗口长度过大可能零溢出或溢出。

   - 实现：用 `rolling.apply(rolling_prod)`。

   - 优化：

     - 对于长度很长的窗口（比如 N>100），直接 `np.prod` 可能会导致数值溢出，建议先对数 sum 再 exp。

       ```python
       return np.exp(df.rolling(window).apply(lambda arr: np.log(arr).sum()))
       ```

     - 如果数据中可能存在非正值，应先剔除为 0 或加一个很小的 epsilon。

5. **delta / delay（差分与滞后）**

   - 功能：分别计算 `df_t - df_{t-period}` 和 `df.shift(period)`。
   - 实现都可以直接调用 Pandas 的 `.diff(period)` 与 `.shift(period)`。
   - 优化：无需改进。

6. **rank（横截面排名百分比）**

   - 功能：在“同一天”的所有股票中，把每只股票的数值转为 0~1 之间的排名分位。
   - 实现：`df.rank(axis=1, pct=True)`。
   - 优化：如果仅想在某一组股票池内做排名，可先用 `df[chosen_universe]` 再做 rank。

7. **scale（L1 归一化）**

   - 功能：把 `df` 每列的绝对值之和缩放到 `k`。
   - 实现：`df.mul(k).div(np.abs(df).sum(axis=0))`。
   - 优化：
     - 如果列中存在 NaN，建议先用 `.fillna(0)` 或相似操作处理。
     - 如果只想对行进行归一化（比如当日所有股票加起来为 1），应把 `axis=1` 作为分母。

8. **ts_argmax / ts_argmin（滚动最大/最小值出现时间）**

   - 功能：计算在过去 N 天内，哪一天最大/最小，以“距离当前的天数”表示（1~N）。
   - 实现：`rolling.apply(lambda arr: np.argmax(arr) + 1)`。
   - 优化：
     - `raw=True`：可提升传入 numpy 数组的效率。
     - 需要注意，`rolling.apply` 对于大数据量会比较慢，因为底层会对每个窗口调用 Python 回调。如果 N 很大或样本很大，需要用 `numba` 重写、或者拆解为分块计算。

9. **decay_linear（线性加权移动平均）**

   - 功能：给最近的数据更高权重，越久远的权重越低。公式相当于窗口内：

     LWMAt=∑i=0N−1(i+1) dft−i∑i=1Ni(i=0表示当日，i=N−1表示N−1天前)  \text{LWMA}_t = \frac{\sum_{i=0}^{N-1} (i+1)\,\text{df}_{t-i}}{\sum_{i=1}^N i} \quad (i=0 表示当日， i=N-1 表示 N-1 天前)

   - 实现有几点需要注意：

     1. 你提供了两次 `decay_linear` 定义，第二次会覆盖第一次，需要合并；
     2. 尽量不要在函数内部用 `inplace=True` 修改原始 `df`，否则调用者有时并不希望原数据被篡改。
     3. 如果直接在 Pandas 上对较大数据运行这个双层 Python 循环，会比较慢。

   - 优化建议：

     - 用 `pandas.Series.ewm` 代替手写的 LWMA ：

       ```python
       # 近似等效：调用指数加权移动平均，但不完全相同
       df.ewm(span=period, adjust=False).mean()
       ```

       如果非要线性权重，可以用 `numpy.lib.stride_tricks.sliding_window_view` + 矩阵乘法来加速，或者引入 `numba`。

     - 如果因子计算只需要对单列/少量列做 LWMA，现有实现可接受；如果要对成百上千只股票做，建议重写成更高效的方式（例如 `numba.jit`）。

   - 下面给出一个用 Pandas 原生 `ewm` 近似、以及一个基于 `numpy.lib.stride_tricks` 的加速示例（仅供参考）：

     ```python
     # 方法一：指数加权移动平均（并非线性权重，但常被用作替代）
     def decay_ewm(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
         """
         用指数加权移动平均（EWMA）近似线性加权效果。
         :param df: pandas DataFrame
         :param span: EWM span
         :return: EWM 结果
         """
         return df.ewm(span=span, adjust=False).mean()
     
     # 方法二：numpy 加速线性加权
     import numpy.lib.stride_tricks as sk
     
     def decay_linear_fast(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
         """
         用 numpy sliding_window_view + 矩阵运算做线性加权平均，加速大规模计算。
         :param df: pandas DataFrame
         :param period: 窗口大小
         :return: DataFrame
         """
         arr = df.to_numpy()  # shape (n_days, n_cols)
         n_days, n_cols = arr.shape
     
         # 首先创建一个 shape=(n_days-period+1, period, n_cols) 的滑窗视图
         # sliding_window_view 需要 numpy >= 1.20
         windows = sk.sliding_window_view(arr, window_shape=(period, n_cols))  # 结果维度需要按列分片
         # 结果 shape = (n_days-period+1, period, n_cols)
         windows = windows[:, :, :]  # 直接使用
     
         # 计算线性权重（长度为 period）
         weights = np.arange(1, period + 1, dtype=np.float64)
         weights = weights / weights.sum()  # 归一化
     
         # 对每个日期批量做加权求和
         # windows 是 (有效天数, period, n_cols)，要对第二维 (period) 做加权
         weighted = np.tensordot(windows, weights, axes=([1], [0]))
         # weighted shape = (n_days-period+1, n_cols)
     
         # 最终结果要拼回成 (n_days, n_cols)，前面的 period-1 行可置 NaN 或原始值
         out = np.full((n_days, n_cols), np.nan)
         out[period-1:] = weighted
     
         return pd.DataFrame(out, index=df.index, columns=df.columns)
     ```

   - 在上述优化示例中，`decay_linear_fast` 会显著加速大规模数据上的线性加权计算，但需要较新版本的 NumPy 才能使用 `sliding_window_view`。

------

## 小结：怎样优化你的代码

1. **去除重复定义**
   - 将 `decay_linear` 两个版本合并为一个，并且不要在函数内部对外部 `df` 做 `inplace=True` 的修改；
   - 读者可以按需用 “Pandas 原生” 或 “NumPy 加速” 版本。
2. **使用 Pandas 原生 Rolling API**
   - 对于求和、均值、标准差、最小/最大、相关/协方差，直接调用 `.rolling(window).sum()/mean()/std()/min()/max()/corr()/cov()`，更加简洁高效；
   - 对于 `ts_rank`、`ts_argmax/ts_argmin` 等需要返回单个值的场景，可用 `rolling.apply`，配合 `raw=True` 提高速度。
3. **避免在函数内修改原始 `DataFrame`**
   - 只有在确实需要“就地”（in-place）修改时，才用 `inplace=True`。通常因子计算流程要保留原始数据，以便后续对比，因此不要修改输入。
4. **数值稳定性与性能**
   - 对于滚动乘积（`product`），如果窗口较大，直接 `np.prod` 可能会数值下溢或上溢，可考虑先用对数变换；
   - 对于滚动相关/协方差，如果窗口较小（N<60），直接用 Pandas 即可；如果要更大规模同时计算多个组合，可考虑用 `numba` 加速自定义函数。
5. **注释与文档化**
   - 上述代码已为每个函数加了 docstring 详细说明输入、输出、用途。
   - 在团队协作中，良好的注释方便后人阅读与维护，也能避免误用。

------

以上便是对你提供的辅助函数的全面解读与优化建议，同时附上标注了优化点与对应注释的代码示例。你可以根据自己的实际需求，选择相应的方法进行因子计算与验证。祝开发顺利！