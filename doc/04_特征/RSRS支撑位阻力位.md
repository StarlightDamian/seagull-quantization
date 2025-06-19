RSRS（**Relative Strength Relative Support**，相对强弱相对支撑位阻力位）是一种技术分析方法，通过对支撑位和阻力位的斜率变化进行量化，来判断市场的趋势和关键价位。它的主要思想是利用回归分析计算股价的相对强弱，以动态方式捕捉支撑和阻力的变化。

### RSRS 的计算步骤

1. **收集历史数据**

   - 获取一段时间内的 **最高价** 和 **最低价** 数据。

2. **回归分析**

   - 使用线性回归将最低价作为自变量 xx，最高价作为因变量 yy。
   - 回归方程为： y=β0+β1xy = \beta_0 + \beta_1 x 其中 β1\beta_1 是回归的斜率，反映支撑与阻力的相对强弱。

3. **标准化斜率**

   - 为便于比较，斜率 

     β1\beta_1

      通常需要标准化处理：

     ——Zβ1=β1−μσZ_{\beta_1} = \frac{\beta_1 - \mu}{\sigma}

     - μ\mu：历史斜率的均值。
     - σ\sigma：历史斜率的标准差。

4. **计算相对强弱指标**

   - 标准化后的斜率 

     Zβ1Z_{\beta_1}

      被用作支撑与阻力位的强弱指标：

     - Zβ1>0Z_{\beta_1} > 0：表示市场处于上升趋势，支撑较强。
     - Zβ1<0Z_{\beta_1} < 0：表示市场处于下降趋势，阻力较强。

5. **策略判断**

   - 根据 

     Zβ1Z_{\beta_1}

      的值可以判断是否满足交易信号：

     - Zβ1Z_{\beta_1} 高于某一阈值（例如 1.0）：买入信号。
     - Zβ1Z_{\beta_1} 低于某一阈值（例如 -1.0）：卖出信号。

------

### Python 实现 RSRS

下面是一个基于 `numpy` 和 `pandas` 的 RSRS 简单实现：

```python
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_rsrs(data, window=14):
    """
    计算 RSRS 相对支撑位阻力位指标
    :param data: DataFrame，必须包含 'high' 和 'low' 列
    :param window: 滑动窗口大小
    :return: DataFrame，新增 'beta' 和 'zscore' 列
    """
    betas = []
    
    for i in range(len(data)):
        if i < window - 1:
            betas.append(np.nan)
        else:
            window_data = data.iloc[i-window+1:i+1]
            slope, intercept, r_value, p_value, std_err = linregress(window_data['low'], window_data['high'])
            betas.append(slope)
    
    data['beta'] = betas
    data['zscore'] = (data['beta'] - np.nanmean(betas)) / np.nanstd(betas)
    
    return data

# 示例数据
data = pd.DataFrame({
    'high': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'low': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
})

result = calculate_rsrs(data, window=5)
print(result[['beta', 'zscore']])
```

------

### 注意事项

1. **参数调整**
   - 窗口大小 windowwindow 会显著影响斜率的平滑程度，需根据标的物特性调节。
   - 阈值选择需结合具体市场测试。
2. **适用范围**
   - RSRS 适合波动性较大的市场，例如股票或商品期货。
3. **回测与验证**
   - 使用历史数据对策略进行回测，验证其有效性。

RSRS 是一种对支撑和阻力进行量化的创新技术，结合其他指标使用效果会更佳。





你的代码计算 `RSRS` 指标相对支撑位阻力位已经比较清晰，针对批量计算和性能优化，以下是一些优化建议。同时，我还会提供如何计算因子信息系数（IC）和召回率的思路。

### 1. **性能优化：**

#### 使用 `rolling` 进行批量计算

目前你是在每个时间步逐个计算 `beta` 值，这样的做法会导致循环效率较低，尤其是在大规模数据上。可以通过 `rolling` 滚动窗口来批量处理，避免逐行计算。

```python
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_rsrs(data, window=14):
    """
    计算 RSRS 相对支撑位阻力位指标
    :param data: DataFrame，必须包含 'high' 和 'low' 列
    :param window: 滑动窗口大小
    :return: DataFrame，新增 'beta' 和 'zscore' 列
    """
    # 使用滚动窗口计算 beta
    betas = data['low'].rolling(window).apply(
        lambda x: linregress(x, data['high'][x.index])[0] if len(x) == window else np.nan, 
        raw=False
    )

    # 添加 beta 和 zscore
    data['beta'] = betas
    data['zscore'] = (data['beta'] - np.nanmean(betas)) / np.nanstd(betas)
    
    return data

# 示例数据
data = pd.DataFrame({
    'high': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'low': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
})

result = calculate_rsrs(data, window=5)
print(result[['beta', 'zscore']])
```

使用 `rolling` 和 `apply` 可以将滚动窗口操作并行化，避免了逐行处理的数据滞后。

### 2. **批量计算：**

如果你的数据集非常大，并且想要在多个 `full_code` 上计算，可以使用 `groupby` 来进行批量计算。对于不同股票代码的处理，可以在 `groupby` 内部应用相同的 `RSRS` 计算函数。

```python
def calculate_rsrs_batch(data, window=14):
    """
    对每个股票代码批量计算 RSRS
    :param data: DataFrame，必须包含 'high', 'low' 和 'full_code' 列
    :param window: 滑动窗口大小
    :return: DataFrame，新增 'beta' 和 'zscore' 列
    """
    def rsrs_single(stock_data):
        return calculate_rsrs(stock_data, window)
    
    return data.groupby('full_code').apply(rsrs_single).reset_index(drop=True)

# 示例数据
data = pd.DataFrame({
    'full_code': ['000001.SZ', '000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ', '000002.SZ'],
    'high': [10, 11, 12, 20, 21, 22],
    'low': [8, 9, 10, 18, 19, 20],
    'date': pd.date_range('2024-01-01', periods=6)
})

result = calculate_rsrs_batch(data, window=3)
print(result[['full_code', 'beta', 'zscore']])
```

### 3. **因子IC计算：**

IC（信息系数）衡量的是因子与未来收益的相关性。为了计算 `RSRS` 因子的 IC 值，你可以根据每个股票的历史 `RSRS` 值与其未来收益之间的皮尔逊相关系数来计算。具体可以通过如下方式计算：

```python
def calculate_ic(data, window=14):
    """
    计算因子 IC（信息系数），这里以 'zscore' 作为因子，未来收益为目标
    :param data: DataFrame，必须包含 'zscore' 和 'close' 列
    :param window: 用于计算未来收益的窗口大小
    :return: IC 值
    """
    data['future_return'] = data['close'].shift(-window) / data['close'] - 1  # 计算未来收益
    ic_values = data.groupby('full_code').apply(
        lambda x: x['zscore'].corr(x['future_return']) if len(x) >= window else np.nan
    )
    return ic_values.mean()

# 示例数据
data = pd.DataFrame({
    'full_code': ['000001.SZ', '000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ', '000002.SZ'],
    'high': [10, 11, 12, 20, 21, 22],
    'low': [8, 9, 10, 18, 19, 20],
    'close': [9, 10, 11, 19, 20, 21],
    'zscore': [0.5, 0.6, 0.7, -0.3, 0.2, 0.1],
    'full_code': ['000001.SZ', '000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ', '000002.SZ'],
})

ic_value = calculate_ic(data, window=3)
print("因子 IC 值:", ic_value)
```

### 4. **召回率计算：**

如果你想计算模型的召回率，通常是需要对因子进行回测或者预测的分类任务（如：买入、卖出、持有）。召回率通常表示正确识别出的正样本与实际正样本的比值。假设你有一个因子预测的方向（如：1 代表买入信号，0 代表卖出信号），并且你能够获得未来实际收益的标签（如：涨停或跌停），你可以根据这些标签来计算召回率。

```python
from sklearn.metrics import recall_score

def calculate_recall(y_true, y_pred):
    """
    计算召回率
    :param y_true: 实际标签，1表示涨停，0表示跌停或其他
    :param y_pred: 预测标签
    :return: 召回率
    """
    return recall_score(y_true, y_pred)

# 示例：假设我们有因子的预测和实际涨跌情况
y_true = [1, 1, 0, 1, 0, 1]  # 实际涨停/跌停标签
y_pred = [1, 0, 0, 1, 1, 0]  # 因子预测标签

recall = calculate_recall(y_true, y_pred)
print("召回率:", recall)
```

### 总结：

-   **批量计算**：通过 `groupby` 和 `apply` 函数对每只股票进行批量计算，可以有效提升计算效率。
-   **性能优化**：通过 `rolling.apply` 来避免逐个循环计算 `beta`，提高计算速度。
-   **IC 和召回率计算**：使用皮尔逊相关系数来计算 `RSRS` 因子的 IC 值，利用真实标签和预测标签计算召回率。

这些优化和方法可以帮助你更好地计算和评估 `RSRS` 因子的效果。如果你的数据量更大，也可以考虑使用 `Dask` 或 `multiprocessing` 来并行化计算。

在这段代码中，`betas` 是通过滚动窗口（rolling window）计算的 `线性回归（Linear Regression）` 的斜率。让我们详细拆解这段代码，并说明 `beta` 的含义及其在量化中的作用。

### 1. **`beta` 在量化中的意义：**

在量化交易中，`beta` 通常用于衡量一个资产（如股票）相对于某个基准（例如，股票指数）的波动性或敏感度。具体来说，`beta` 表示的是资产的价格与基准价格之间的线性关系中的斜率。在这里，`beta` 是通过 **线性回归** 来计算的，衡量的是 `low`（最低价）与 `high`（最高价）之间的关系。

-   `beta` 趋势解释：
    -   如果 `beta` 为正，意味着 `low` 和 `high` 之间存在正相关，最低价上涨时，最高价也可能会跟着上涨。
    -   如果 `beta` 为负，意味着 `low` 和 `high` 之间的关系为负相关，最低价上涨时，最高价可能下跌。
    -   `beta` 的绝对值越大，表示 `low` 和 `high` 之间的关系越强。

### 2. **这段代码的作用：**

```python
betas = data['low'].rolling(window).apply(
    lambda x: linregress(x, data['high'][x.index])[0] if len(x) == window else np.nan, 
    raw=False
)
```

这段代码的作用是 **使用滚动窗口** 对每个窗口（由 `window` 参数决定的时间区间）内的 `low`（最低价）与 `high`（最高价）数据进行线性回归，并计算回归的斜率（即 `beta`）。具体步骤如下：

-   **`rolling(window)`**：这会为数据创建一个滚动窗口，窗口大小为 `window`，即每次计算都涉及 `window` 个时间点的数据。在量化中，`window` 通常是一个固定的时间跨度，比如 14 天或者 20 天等。

-   **`apply(lambda x: ...)`**：`apply` 方法用于对每个窗口内的数据应用一个函数，这里用的是一个匿名函数 `lambda x`。对于每个窗口（即每个时间步的过去 `window` 天的数据）：

    -   `linregress(x, data['high'][x.index])`

        ：这是调用 

        ```
        scipy.stats.linregress
        ```

         函数对窗口中的 

        ```
        low
        ```

        （最低价）和 

        ```
        high
        ```

        （最高价）进行线性回归。

        -   `x` 是窗口中的 `low` 数据。
        -   `data['high'][x.index]` 是对应时间点的 `high` 数据，`x.index` 确保了这两个系列的时间对齐。
        -   `linregress` 函数返回一组统计量，其中第一个值 `linregress(...)[0]` 是回归的斜率（即 `beta`），它表示 `low` 和 `high` 之间的线性关系的强度和方向。

-   **`if len(x) == window else np.nan`**：由于滚动窗口在开始的几个数据点时没有足够的历史数据，这段代码确保了只有在窗口内有足够数据时（即窗口长度为 `window`）才计算 `beta`，否则返回 `np.nan`。

-   **`raw=False`**：这意味着 `apply` 会将 `x` 作为一个 `Series` 对象传入，而不是一个原始的 `ndarray`。这对于处理时间序列数据是有用的，因为我们需要保留时间索引。

### 3. **总结与作用：**

-   **`betas` 的含义：**
    -   `beta` 反映了某只股票的最低价（`low`）与最高价（`high`）之间的线性关系的斜率。
    -   在量化交易中，这个 `beta` 可以用来衡量某只股票的波动性（尤其是它如何随着最低价的变化而变化）。
    -   `beta` 的值越大，表示 `low` 和 `high` 之间的关系越强，通常也意味着股票的波动性较大。
-   **作用：**
    -   这段代码的作用是计算每个时间点（从 `window` 大小的起始位置开始），以过去 `window` 天的数据为窗口，计算出 `low` 和 `high` 之间的线性回归斜率（即 `beta`）。这些 `beta` 值随后可以作为一个因子进行进一步分析，例如用于生成交易信号或为风险模型提供输入。
    -   **`zscore`** 是对 `beta` 值的标准化处理，能够衡量 `beta` 的相对位置，通常用于判断当前 `beta` 的极端程度。计算出来的 `zscore` 可以帮助识别价格的极端波动或回归现象。

### 4. **应用场景：**

-   `RSRS` 指标通常用于 **支撑位和阻力位的判断**。你可以通过 `beta` 来判断价格走势的斜率，并利用 `zscore` 来判断当前的价格是否偏离了某个“标准”区间。
-   在技术分析中，`RSRS` 因子可以用来发现股价是否即将发生回撤或突破。