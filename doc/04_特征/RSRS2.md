嗯，用户想用RSRS指标来买卖ADA-USD和ETH-USD这两个加密货币。首先，我需要回想一下RSRS是什么，以及如何在Python中实现它。RSRS是阻力支撑相对强度指标，由光大证券提出，主要是通过线性回归来计算支撑线和阻力线的斜率，判断市场的强弱。



用户已经导入了vectorbt库，这可能意味着他们希望使用这个库来进行回测和信号生成。VectorBT是一个高效的量化分析库，支持并行计算，适合处理多个标的和大量数据。



首先，我需要确定RSRS的计算方法。通常，RSRS的计算步骤包括：选择一定的时间窗口（比如18天），对最高价和最低价进行线性回归，得到斜率作为阻力支撑强度的指标。然后可能会有标准化的步骤，比如计算Z-score，或者结合右偏修正等方法。此外，可能需要结合移动平均或者其他指标来生成交易信号。



接下来，用户可能希望根据RSRS的数值生成买入和卖出信号。常见的策略是当RSRS超过某个阈值时买入，低于另一个阈值时卖出。或者结合其移动平均线，比如当RSRS上穿其移动平均线时买入，下穿时卖出。



然后，需要考虑如何处理多个标的。VectorBT可以同时处理多个symbols，所以需要为每个symbol分别计算RSRS指标，并生成各自的交易信号。在下载数据时，可能需要获取历史价格数据，包括高、低、收盘价等。



现在，具体步骤可能是这样的：



1.   下载数据：使用vectorbt的YahooFinance数据模块，或者别的数据源，获取ADA-USD和ETH-USD的历史数据，包括high、low、close。



2.   计算RSRS指标：对每个symbol，按窗口计算最高价和最低价的线性回归斜率。比如，用n天的时间窗口，对每日的最高价和最低价进行回归，得到斜率作为当天的RSRS值。



3.   可能需要对RSRS进行标准化或平滑处理，比如计算Z-score，或者使用移动平均线来生成信号。



4.   生成交易信号：比如当RSRS上穿某个阈值（如1）时买入，下穿另一个阈值（如-1）时卖出。或者结合其移动平均线，形成金叉死叉。



5.   使用vectorbt组合信号，进行回测，评估策略表现。



需要注意的是，不同的参数设置可能会影响策略的效果，比如窗口长度、阈值的选择，这些可能需要优化。但用户可能只是需要一个基本的实现框架，所以先提供基础代码，再说明参数调整的可能性。



另外，vectorbt的用法可能需要查阅文档，比如如何为多个symbol处理指标，如何生成买卖信号，以及如何构建投资组合。例如，使用vbt.YFData.download(symbols)来获取数据，然后为每个symbol计算RSRS，生成entries和exits信号，最后通过Portfolio.from_signals创建组合。



在编写代码时，需要注意数据对齐问题，确保每个symbol的指标计算正确。例如，对于每个symbol，单独计算RSRS，然后生成各自的信号。VectorBT应该能够自动处理多个资产的情况，只要数据是正确结构的。



可能的代码结构如下：



1.   下载数据：

data = vbt.YFData.download(symbols, start='2019-01-01', end='2023-01-01').get('Close')



不过可能需要high和low数据来计算RSRS，所以需要获取完整的OHLC数据。因此可能需要调整data的获取方式，比如：



price = vbt.YFData.download(symbols, start='2019-01-01', end='2023-01-01')

highs = price.high

lows = price.low

closes = price.close



然后对highs和lows进行逐symbol的处理。



2.   定义计算RSRS的函数。例如，对于每个symbol的high和low序列，滚动计算线性回归斜率。可以使用vbt的rolling_apply，或者更高效的方法。例如，对于每个窗口内的最高价和最低价，进行线性回归，其中low作为自变量X，high作为因变量Y，斜率即为RSRS值。



不过通常RSRS是将每日的最高价和最低价分别作为Y和X，进行回归，求斜率。比如，在每个滚动窗口内，取每日的低价作为X，高价作为Y，进行OLS回归，得到斜率。这可能需要为每个窗口计算斜率。



在Python中，可以使用numpy的polyfit函数，或者scipy的linregress。不过对于滚动应用，需要高效的方法。VectorBT可能提供了相关的函数，例如使用vbt.IndicatorFactory来创建自定义指标。



或者，手动实现滚动线性回归。例如：



def calculate_rsrs(high, low, window=18):

slopes = np.full_like(high, np.nan)

for i in range(window, len(high)):

x = low[i-window+1:i+1]

y = high[i-window+1:i+1]

\# 添加常数项，进行回归

\# 这里可能需要注意，是使用low作为自变量，high作为因变量

\# 根据RSRS的定义，可能正确的方式是low作为X，high作为Y

slope, intercept = np.polyfit(x, y, 1)

slopes[i] = slope

return slopes



然后将这个函数应用到每个symbol的high和low序列上。



不过使用循环可能效率不高，特别是在处理多个symbol时。可以考虑向量化的方法，或者利用vectorbt的rolling窗口功能。



或者，使用vectorbt的Rolling窗口，结合apply函数。例如：



n = 18

rsrs = vbt.rolling_apply(

window=n,

apply_func=lambda high, low: np.polyfit(low, high, 1)[0],

high=highs,

low=lows,

window_len=n,

minp=n,

\# 可能需要处理输入参数，这里可能需要调整，因为rolling_apply可能传递的是窗口数组

)



不过，这里可能需要自定义一个指标，因为rolling_apply的apply_func通常接收的是每个窗口的数组。例如，对于每个时间点，取出当前窗口的high和 low，进行回归。这时候可能需要重新设计apply_func的参数。



或者，可以分开处理每个symbol，因为对于每个symbol来说，high和low是独立的。因此，可能需要遍历每个symbol，计算其对应的RSRS，然后将结果合并。



例如：



rsrs = {}

for sym in symbols:

high = highs[sym]

low = lows[sym]

\# 计算每个symbol的RSRS

rsrs_sym = high.rolling(window=n).apply(lambda h: np.polyfit(low.loc[h.index], h, 1)[0], raw=False)

rsrs[sym] = rsrs_sym



不过这可能会有问题，因为在rolling.apply中，传递的h是窗口内的high值，但需要对应的low值。可能需要同时访问high和low的窗口数据，这可能需要更复杂的处理。例如，可以用一个自定义函数，接受索引位置，然后获取对应的low和 high数据。



或者，使用一个高阶函数，将low序列作为闭包变量。例如：



def get_rsrs_calculator(low):

def calculator(high_window):

\# 这里的high_window是一个窗口的high数据

\# 对应的low数据需要根据索引来获取

\# 但可能无法直接获取对应的low数据，因为索引可能不明确？

\# 所以这种方法可能有问题。

\# 或者，假设high和 low的索引是同步的，即每个high_window对应的low_window是同一时间窗口的low数据

low_window = low.loc[high_window.index]

slope = np.polyfit(low_window, high_window, 1)[0]

return slope

return calculator



然后对每个symbol：



rsrs_list = []

for sym in symbols:

high = highs[sym]

low_series = lows[sym]

calculator = get_rsrs_calculator(low_series)

rsrs_sym = high.rolling(window=n).apply(calculator, raw=False)

rsrs_list.append(rsrs_sym)



rsrs = pd.concat(rsrs_list, axis=1)

rsrs.columns = symbols



这可能可行，但需要测试是否正确。不过，在rolling.apply中传递的是窗口数据，其索引是正确的，所以low_series.loc[window.index]应该能获取到对应的low数据。



不过这可能效率不高，尤其当数据量大时。或者，可以采用向量化的方式来计算滚动回归。例如，使用线性代数的方法，快速计算每个窗口的斜率。



另外，可以使用numpy的sliding_window_view结合polyfit，但这可能比较复杂。



假设我们能够正确计算出每个symbol的RSRS，那么接下来就是生成信号。例如，当RSRS大于某个阈值时买入，小于另一个阈值时卖出。或者结合移动平均。



例如，计算RSRS的Z-score，当Z-score超过1时买入，低于-1时卖出。或者使用RSRS的移动平均线，比如当RSRS上穿其MA时买入，下穿时卖出。



例如：



\# 计算RSRS的Z-score

rsrs_mean = rsrs.rolling(window=252).mean()

rsrs_std = rsrs.rolling(window=252).std()

rsrs_z = (rsrs - rsrs_mean) / rsrs_std



entries = rsrs_z > 1  # 买入信号

exits = rsrs_z < -0.5  # 卖出信号



或者，使用原始RSRS值，比如当RSRS大于某个值时买入，小于另一个值时卖出。



或者，结合右偏修正，比如RSRS指标还有另一种计算方式，即考虑标准分和右偏分，但可能用户只需要基础版本。



接下来，使用vectorbt的Portfolio.from_signals方法，根据entries和exits生成交易，并计算绩效。



例如：



portfolio = vbt.Portfolio.from_signals(

closes=closes,

entries=entries,

exits=exits,

fees=0.001,  # 假设交易费率0.1%

freq='1D'

)



然后可以查看绩效：portfolio.stats()



但需要注意，entries和exits的结构需要与closes对齐，即每个symbol对应的列是否正确。



此外，可能需要考虑做多和做空的策略，但用户的问题可能只是简单的买入卖出，所以可能entries代表开多仓，exits代表平仓。



但用户的问题描述是“买入卖出”，所以可能策略是当RSRS产生买入信号时买入，卖出信号时卖出。但需要明确是连续持有还是每次信号都执行交易。例如，每次买入信号时买入，卖出信号时卖出，可能产生多次交易。



或者，可能需要在买入信号出现时买入，直到卖出信号出现时平仓，期间只持有多头头寸。这可以通过entries和exits的布尔型Series来实现，其中entry是信号出现的位置，exit同理，vectorbt会自动处理仓位的变化。



现在，将这些步骤整合到代码中。可能需要处理各个symbol的RSRS计算，然后生成对应的entries和exits信号。



可能的完整代码示例：



import vectorbt as vbt

import numpy as np

import pandas as pd



symbols = ["ADA-USD", "ETH-USD"]



\# 下载历史数据

data = vbt.YFData.download(

symbols,

start='2020-01-01',

end='2023-01-01'

)

highs = data.high

lows = data.low

closes = data.close



\# 计算RSRS

n = 18  # 窗口大小

rsrs = pd.DataFrame(index=highs.index, columns=symbols)



for sym in symbols:

high = highs[sym]

low = lows[sym]

\# 对每个symbol计算RSRS

for i in range(n, len(high)):

window_high = high.iloc[i-n:i]

window_low = low.iloc[i-n:i]

if len(window_high) < n:

continue

\# 进行线性回归，high = slope * low + intercept

slope, intercept = np.polyfit(window_low, window_high, 1)

rsrs.loc[window_high.index[-1], sym] = slope



rsrs = rsrs.astype(float)  # 确保数据类型正确



\# 生成信号：假设RSRS的Z-score超过1时买入，低于-1时卖出

zscore_window = 252  # 一年的交易日

rsrs_mean = rsrs.rolling(zscore_window).mean()

rsrs_std = rsrs.rolling(zscore_window).std()

rsrs_z = (rsrs - rsrs_mean) / rsrs_std



entries = rsrs_z > 1

exits = rsrs_z < -1



\# 创建投资组合

portfolio = vbt.Portfolio.from_signals(

closes=closes,

entries=entries,

exits=exits,

fees=0.001,  # 0.1%的交易费用

freq='1D'

)



\# 输出绩效报告

print(portfolio.stats())



这段代码可能存在一些问题，比如在计算RSRS时，使用双重循环效率较低，尤其是当数据量较大时。为了提高效率，可以尝试优化RSRS的计算，例如使用滚动窗口的向量化操作。



另外，在计算线性回归时，如果窗口内的数据存在NaN，可能会出错。需要确保在计算时窗口内的数据是完整的。



或者，可以使用vbt的IndicatorFactory来创建自定义指标，从而更高效地计算RSRS。例如：



from vectorbt.indicators.factory import IndicatorFactory



def rsrs_indicator(high, low, window=18):

\# 定义指标的计算逻辑

def calculate(window_high, window_low):

slope = np.polyfit(window_low, window_high, 1)[0]

return slope



\# 使用rolling apply

return high.vbt.rolling_apply(

window,

lambda x: calculate(x, low.loc[x.index]),

window_len=window,

minp=window,

sym_names=['high', 'low']

)



然后对每个symbol应用这个指标：



rsrs = vbt.merge(

rsrs_indicator(highs[sym], lows[sym], window=n) for sym in symbols

).rename(columns=lambda x: symbols[x])



这可能更高效，但需要验证是否正确。



此外，用户可能需要调整参数，例如窗口长度n，Z-score的窗口，以及买卖阈值。可能需要参数优化来找到最佳设置。



最后，用户可能需要可视化结果，例如绘制RSRS指标、买卖信号，以及资产净值曲线。可以使用vectorbt内置的绘图功能，例如portfolio.plot()。

以下是使用VectorBT和RSRS指标进行买卖的步骤和代码示例：

### 步骤1：获取历史数据
```python
import vectorbt as vbt
import numpy as np

symbols = ["ADA-USD", "ETH-USD"]

# 下载历史数据（包含高、低、收盘价）
data = vbt.YFData.download(
    symbols,
    start='2020-01-01',
    end='2023-01-01'
)
highs = data.high  # 最高价序列
lows = data.low    # 最低价序列
closes = data.close # 收盘价序列
```

### 步骤2：计算RSRS指标
```python
# 定义RSRS计算函数
def calculate_rsrs(high, low, window=18):
    rsrs = np.full_like(high, np.nan)
    for i in range(window, len(high)):
        x = low[i-window:i]  # 低点作为自变量X
        y = high[i-window:i] # 高点作为因变量Y
        if len(x) < 2 or len(y) < 2:
            continue
        slope = np.polyfit(x, y, 1)[0]  # 线性回归斜率
        rsrs[i] = slope
    return rsrs

# 为每个标的计算RSRS
rsrs = vbt.merge([
    highs[sym].vbt.rolling_apply(
        window=18,
        apply_func=lambda h: calculate_rsrs(h, lows[sym].loc[h.index], window=18)[-1],
        window_len=18,
        minp=18
    ).rename(sym)
    for sym in symbols
], axis=1)
```

### 步骤3：生成交易信号
```python
# 标准化RSRS（Z-Score）
zscore_window = 252  # 一年的交易日
rsrs_mean = rsrs.rolling(zscore_window).mean()
rsrs_std = rsrs.rolling(zscore_window).std()
rsrs_z = (rsrs - rsrs_mean) / rsrs_std

# 定义买卖信号（Z > 1买入，Z < -1卖出）
entries = rsrs_z > 1   # 买入信号
exits = rsrs_z < -1    # 卖出信号
```

### 步骤4：执行回测
```python
# 构建投资组合
portfolio = vbt.Portfolio.from_signals(
    closes=closes,      # 收盘价作为交易价格
    entries=entries,    # 买入信号
    exits=exits,        # 卖出信号
    fees=0.001,         # 交易费率0.1%
    freq='1D'           # 日频数据
)

# 输出绩效报告
print(portfolio.stats())
```

### 步骤5：可视化结果（可选）
```python
# 绘制净值曲线
portfolio.value().vbt.plot()
```

### 参数说明
1. **窗口长度（`window=18`）**：RSRS计算的历史窗口，通常取18天。
2. **Z-Score窗口（`zscore_window=252`）**：用于标准化RSRS的滚动窗口，通常取一年交易日。
3. **交易阈值（Z > 1买入，Z < -1卖出）**：可根据历史数据优化调整。

### 注意事项
- **数据对齐**：确保`highs`、`lows`和`closes`的索引对齐。
- **计算效率**：循环计算RSRS可能较慢，建议对数据进行预处理。
- **参数优化**：使用`vectorbt`的`gridsearch`功能优化窗口和阈值参数。

通过调整上述参数和逻辑（例如添加移动平均过滤），您可以进一步优化策略表现。