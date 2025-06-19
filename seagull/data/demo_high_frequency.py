# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:23:26 2025

@author: awei

demo_high_frequency

针对5000只股票6年的5分钟K线数据，高效计算多个技术指标需要结合 **向量化计算**、**并行处理** 和 **内存优化**。以下是分步实现方案及代码示例：

---

### **1. 数据结构优化**
假设原始数据格式为 **长表（Long Format）**，结构如下：
- 索引：无（或时间戳）
- 列：`symbol`（股票代码）、`datetime`（时间戳）、`open`、`high`、`low`、`close`、`volume`

**优化步骤**：
1. **按股票分组存储**：使用 `parquet` 格式按股票代码分块存储，减少内存压力。
2. **按时间排序**：确保每个股票的数据按时间升序排列。
3. **使用高效数据类型**：如 `float32` 替代 `float64`，`category` 类型存储股票代码。

```python
import pandas as pd
import numpy as np

# 示例数据加载（假设已按symbol分块存储）
df = pd.read_parquet("kline_data.parquet")
df['symbol'] = df['symbol'].astype('category')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['symbol', 'datetime'])
```

---

### **2. 技术指标计算公式**
#### **(1) 流动性（Liquidity）**
- **定义**：Amihud流动性指标（单位成交量的价格冲击）  
  \[
  \text{Liquidity} = \frac{|\text{close}_t - \text{close}_{t-1}|}{\text{volume}_t}
  \]
- **优化**：避免零成交量导致除零错误。

```python
df['returns'] = df.groupby('symbol')['close'].pct_change()
df['liquidity'] = np.abs(df['returns']) / df['volume'].replace(0, np.nan)
```

#### **(2) 动量反转（Momentum & Reversal）**
- **动量**：过去N个周期的收益率（示例N=12，即1小时动量）
- **反转**：动量的负值

```python
n_periods = 12  # 1小时窗口（5分钟×12）
df['momentum'] = df.groupby('symbol')['close'].pct_change(n_periods)
df['reversal'] = -df['momentum']
```

#### **(3) 筹码分布（Chip Distribution）**
- **定义**：价格在滚动窗口内的成交量加权分布，取标准差衡量分散度
  \[
  \text{Chips} = \text{std}(\text{close} \times \text{volume})
  \]

```python
window_chips = 20  # 100分钟窗口
df['price_volume'] = df['close'] * df['volume']
df['chips'] = df.groupby('symbol')['price_volume'].rolling(window_chips).std().values
```

#### **(4) 拥挤度（Crowding）**
- **定义**：当前成交量与过去M日平均成交量的比值（示例M=5日）
  \[
  \text{Crowding} = \frac{\text{volume}_t}{\text{MA}(\text{volume}, M)}
  \]

```python
window_crowd = 5 * 12 * 24  # 5日窗口（假设每日12小时×24个5分钟）
df['volume_ma'] = df.groupby('symbol')['volume'].transform(
    lambda x: x.rolling(window_crowd, min_periods=1).mean()
)
df['crowding'] = df['volume'] / df['volume_ma']
```

#### **(5) 波动率（Volatility）**
- **定义**：滚动窗口内收益率的标准差
  \[
  \text{Volatility} = \text{std}(\text{returns}, N)
  \]

```python
window_vol = 20  # 100分钟窗口
df['volatility'] = df.groupby('symbol')['returns'].rolling(window_vol).std().values
```

#### **(6) 价量相关性（Price-Volume Correlation）**
- **定义**：滚动窗口内价格与成交量的Pearson相关系数

```python
window_corr = 20  # 100分钟窗口

def corr(df):
    return df['close'].corr(df['volume'])

df['corr'] = df.groupby('symbol').rolling(window_corr).apply(corr).values
```

"""

import os
import argparse

import numpy as np
import pandas as pd
# from joblib import Parallel, delayed

from __init__ import path
from utils import utils_database, utils_log, utils_data, utils_character, utils_thread, utils_time
from data import utils_api_baostock

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


def calculate_liquidity(day_df):
    # 流动性
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df['liquidity'] = np.abs(df['returns']) / df['volume'].replace(0, np.nan)

def calculate_high_frequency(day_df, buy_pct=[10, 20], sell_pct=[80, 90]):
    day_df = day_df.reset_index(drop=True)
    total_volume = day_df['volume'].sum()
    ohlc_pct_columns = []
    
    # 计算买入价格（低价百分位）
    day_df = day_df.sort_values('low')
    day_df['low_cum_volume'] = day_df['volume'].cumsum()
    
    for buy_pct_1 in buy_pct:
        # 使用searchsorted高效查找百分位的位置,计算相应的买入
        buy_index = np.searchsorted(day_df['low_cum_volume'], (buy_pct_1 / 100) * total_volume)
        # Ensure buy_index is within bounds
        if buy_index < len(day_df):
            pct_low = f'_{buy_pct_1}pct_5min_low'
            day_df[pct_low] = day_df.iloc[buy_index]['low']
            ohlc_pct_columns.append(pct_low)
        else:
            # Handle case where index is out of bounds, you can choose to set NaN or the last valid value
            day_df[f'_{buy_pct_1}pct_5min_low'] = np.nan
            ohlc_pct_columns.append(f'_{buy_pct_1}pct_5min_low')
        
    # 计算卖出价格（高价百分位）
    day_df = day_df.sort_values('high', ascending=False)
    day_df['high_cum_volume'] = day_df['volume'].cumsum()
    
    for sell_pct_1 in sell_pct:
        sell_index = np.searchsorted(day_df['high_cum_volume'], (1 - (sell_pct_1 / 100)) * total_volume)
        # Ensure sell_index is within bounds
        if sell_index < len(day_df):
            pct_high = f'_{sell_pct_1}pct_5min_high'
            day_df[pct_high] = day_df.iloc[sell_index]['high']
            ohlc_pct_columns.append(pct_high)
        else:
            # Handle case where index is out of bounds, you can choose to set NaN or the last valid value
            day_df[f'_{sell_pct_1}pct_5min_high'] = np.nan
            ohlc_pct_columns.append(f'_{sell_pct_1}pct_5min_high')
        
    # vwap
    day_df['_5min_vwap'] = (((day_df.high + day_df.low + day_df.close) / 3) * day_df.volume).sum() / total_volume
    day_df['_5min_vwap'] = day_df['_5min_vwap'].round(4)
    
    day_df = day_df[['date', 'full_code', '_5min_vwap', 'primary_key', 'freq', 'adj_type'] + ohlc_pct_columns].head(1)
    return day_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2019-01-01', help='When to start high frequency')
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='When to start high frequency')
    parser.add_argument('--date_end', type=str, default='2025-02-23', help='End time for high frequency')
    args = parser.parse_args()
    
    logger.info(f"""task: dwd_feat_incr_high_frequency_5minute
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")
    with utils_database.engine_conn('postgre') as conn:
        high_frequency_df = pd.read_sql(f"""
                    SELECT
                        primary_key
                        ,_5min_vwap
                        ,_10pct_5min_low
                        ,_20pct_5min_low
                        ,_80pct_5min_high
                        ,_90pct_5min_high
                    FROM
                        dwd_feat_incr_high_frequency_5minute
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'
                        """, con=conn.engine)
                        
                        
    high_frequency_df[~((high_frequency_df._5min_vwap==0)|
                      (high_frequency_df._10pct_5min_low==0)|
                      (high_frequency_df._20pct_5min_low==0)|
                      (high_frequency_df._80pct_5min_high==0)|
                      (high_frequency_df._90pct_5min_high==0))]
    #with utils_database.engine_conn('postgre') as conn:
        #pd.read_sql("dwd_feat_incr_high_frequency_5minute", con=conn.engine)
# =============================================================================
#     raw_df = pd.read_csv(f'{path}/data/asset_5min_df.csv')
#     df = raw_df[(raw_df['date']=='2023-03-28')&(raw_df['full_code']=='600183.sh')]
#     
#     # 流动性
#     df['returns'] = df.groupby('symbol')['close'].pct_change()
#     df['liquidity'] = np.abs(df['returns']) / df['volume'].replace(0, np.nan)
#     
#     n_periods = 12
#     df['momentum'] = df['close'].pct_change(n_periods)
#     df['reversal'] = -df['momentum']
# =============================================================================
    #asset_5min_df = pd.read_csv(f'{path}/data/asset_5min_df.csv')

    #asset_5min_df.to_csv(f'{path}/data/asset_5min_df.csv',index=False)
    #asset_5min_df = pd.read_csv(f'{path}/data/asset_5min_df.csv')
    #pipeline(asset_5min_df)