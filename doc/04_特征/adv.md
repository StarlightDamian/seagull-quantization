**`adv{d}`**（即过去 **d** 天的平均日成交额）通常指的是过去 **d** 天内的**日成交额**（成交量 × 收盘价）之平均值。

### **公式**：

advd=1d∑i=1dDaily Dollar Volumei\text{adv}_{d} = \frac{1}{d} \sum_{i=1}^{d} \text{Daily Dollar Volume}_i

其中，**Daily Dollar Volume** 是单日的成交金额，通常由以下公式计算：

Daily Dollar Volumei=Volumei×Close Pricei\text{Daily Dollar Volume}_i = \text{Volume}_i \times \text{Close Price}_i

因此，**`adv{d}`** 也可以写作：

advd=1d∑i=1d(Volumei×Close Pricei)\text{adv}_{d} = \frac{1}{d} \sum_{i=1}^{d} (\text{Volume}_i \times \text{Close Price}_i)

### **解释**：

-   **Volume**：表示当天的成交量。
-   **Close Price**：表示当天的收盘价。
-   **d**：过去的天数，通常取值为 20、30 或 60 天。

### **步骤**：

1.  对于每一天，计算 **Daily Dollar Volume**（即 `成交量 × 收盘价`）。
2.  将过去 **d** 天的 **Daily Dollar Volume** 求和。
3.  将求和结果除以 **d**，得到过去 **d** 天的平均日成交额（**adv{d}**）。

### **举例**：

假设我们有过去 5 天的成交量和收盘价数据：

| 日期       | 成交量 (Volume) | 收盘价 (Close Price) | 日成交额 (Daily Dollar Volume) |
| ---------- | --------------- | -------------------- | ------------------------------ |
| 2023-12-01 | 100,000         | 10                   | 1,000,000                      |
| 2023-12-02 | 150,000         | 9                    | 1,350,000                      |
| 2023-12-03 | 120,000         | 8                    | 960,000                        |
| 2023-12-04 | 180,000         | 11                   | 1,980,000                      |
| 2023-12-05 | 200,000         | 12                   | 2,400,000                      |

那么 **`adv{5}`** 就是过去 5 天的日成交额的平均值：

adv5=1,000,000+1,350,000+960,000+1,980,000+2,400,0005=7,690,0005=1,538,000\text{adv}_{5} = \frac{1,000,000 + 1,350,000 + 960,000 + 1,980,000 + 2,400,000}{5} = \frac{7,690,000}{5} = 1,538,000

因此，过去 5 天的平均日成交额 **`adv{5}`** 为 1,538,000。

### **Python 实现**：

如果你在使用 **Pandas** 来处理股票数据，可以使用以下代码来计算 **`adv{d}`**：

```python
import pandas as pd

# 示例数据
data = {
    'date': ['2023-12-01', '2023-12-02', '2023-12-03', '2023-12-04', '2023-12-05'],
    'volume': [100000, 150000, 120000, 180000, 200000],
    'close': [10, 9, 8, 11, 12]
}

# 创建 DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# 计算每日的成交额
df['daily_dollar_volume'] = df['volume'] * df['close']

# 设置天数 (例如过去 5 天)
d = 5

# 计算过去 d 天的平均日成交额 (adv{d})
df['adv_5'] = df['daily_dollar_volume'].rolling(window=d).mean()

# 查看结果
print(df[['date', 'daily_dollar_volume', 'adv_5']])
```

### **输出示例**：

```
         date  daily_dollar_volume     adv_5
0  2023-12-01              1000000       NaN
1  2023-12-02              1350000       NaN
2  2023-12-03               960000       NaN
3  2023-12-04              1980000       NaN
4  2023-12-05              2400000  1538000.0
```

由于 **`rolling(window=d)`** 只会在数据足够时才开始计算滚动平均值，因此在前 **d-1** 天，`adv{d}` 会是 `NaN`。