为了实现你的目标，即预测每日能够成交5%量的最低买入价（累计成交量5%分位价格）和最高卖出价（累计成交量95%分位价格），我们需要以下步骤：

---

### **1. 数据预处理：生成标签（目标变量）**
首先需从5分钟K线数据中计算每日的 **5%成交量分位价格** 和 **95%成交量分位价格**。

```python
import pandas as pd
import numpy as np

def calculate_volume_quantile_prices(df_5min):
    """
    输入: 5分钟K线数据，列为 ['open', 'high', 'low', 'close', 'volume']
    输出: 每日的5%和95%成交量分位价格
    """
    # 按日期分组，处理每一天的数据
    daily_data = df_5min.groupby(pd.Grouper(freq='D'))
    
    results = []
    for date, day_df in daily_data:
        if day_df.empty:
            continue
        
        # 按价格排序并计算累计成交量
        sorted_low = day_df.sort_values('low')
        sorted_low['cum_volume'] = sorted_low['volume'].cumsum()
        total_volume = sorted_low['volume'].sum()
        
        # 找到5%分位价格（买入）
        buy_5pct = sorted_low[sorted_low['cum_volume'] >= 0.05 * total_volume].iloc[0]['low']
        
        # 按价格倒序排序（卖出）
        sorted_high = day_df.sort_values('high', ascending=False)
        sorted_high['cum_volume'] = sorted_high['volume'].cumsum()
        
        # 找到95%分位价格（卖出）
        sell_95pct = sorted_high[sorted_high['cum_volume'] >= 0.05 * total_volume].iloc[0]['high']
        
        results.append({
            'date': date,
            'buy_5pct_price': buy_5pct,
            'sell_95pct_price': sell_95pct
        })
    
    return pd.DataFrame(results).set_index('date')

# 示例数据构造（假设df_5min的索引为DatetimeIndex）
df_5min = pd.read_csv('5min_data.csv', parse_dates=['timestamp'], index_col='timestamp')
labels_df = calculate_volume_quantile_prices(df_5min)
```

---

### **2. 特征工程：构建预测特征**
基于5分钟K线数据生成每日的特征，例如：

```python
def build_features(df_5min, lookback_days=5):
    """
    构建每日特征：
    - 过去N日的波动率、成交量均值、价格动量等
    - 日内特征：如振幅、VWAP（成交量加权平均价）
    """
    # 按日聚合计算基础指标
    daily_open = df_5min.resample('D').first()['open']
    daily_high = df_5min.resample('D').max()['high']
    daily_low = df_5min.resample('D').min()['low']
    daily_close = df_5min.resample('D').last()['close']
    daily_volume = df_5min.resample('D').sum()['volume']
    
    # 计算VWAP（成交量加权平均价）
    df_5min['vwap'] = (df_5min['volume'] * (df_5min['high'] + df_5min['low'] + df_5min['close']) / 3).cumsum() / df_5min['volume'].cumsum()
    daily_vwap = df_5min.resample('D').last()['vwap']
    
    # 合并基础特征
    features = pd.DataFrame({
        'open': daily_open,
        'high': daily_high,
        'low': daily_low,
        'close': daily_close,
        'volume': daily_volume,
        'vwap': daily_vwap
    })
    
    # 添加滚动窗口特征
    for window in [3, 5, 10]:
        features[f'volatility_{window}d'] = daily_close.pct_change().rolling(window).std()
        features[f'volume_ma_{window}d'] = daily_volume.rolling(window).mean()
    
    # 添加滞后特征
    for lag in [1, 2, 3]:
        features[f'close_lag{lag}'] = daily_close.shift(lag)
    
    return features.dropna()

features_df = build_features(df_5min)
```

---

### **3. 合并特征和标签**
将特征与标签对齐，确保时间一致性：

```python
# 合并特征和标签（确保日期索引一致）
data = pd.merge(features_df, labels_df, left_index=True, right_index=True)
```

---

### **4. 训练LightGBM模型**
使用回归模型预测两个目标变量（买入价和卖出价）：

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 定义特征和目标
X = data.drop(['buy_5pct_price', 'sell_95pct_price'], axis=1)
y_buy = data['buy_5pct_price']
y_sell = data['sell_95pct_price']

# 划分训练集和测试集
X_train, X_test, y_buy_train, y_buy_test = train_test_split(X, y_buy, test_size=0.2, shuffle=False)
_, _, y_sell_train, y_sell_test = train_test_split(X, y_sell, test_size=0.2, shuffle=False)

# 训练买入价模型
model_buy = lgb.LGBMRegressor()
model_buy.fit(X_train, y_buy_train)
y_buy_pred = model_buy.predict(X_test)
print("Buy Price MSE:", mean_squared_error(y_buy_test, y_buy_pred))

# 训练卖出价模型
model_sell = lgb.LGBMRegressor()
model_sell.fit(X_train, y_sell_train)
y_sell_pred = model_sell.predict(X_test)
print("Sell Price MSE:", mean_squared_error(y_sell_test, y_sell_pred))
```

---

### **5. 模型优化与解释**
- **参数调优**：使用 `GridSearchCV` 或 `Optuna` 优化超参数。
- **特征重要性**：分析模型的特征重要性，去除冗余特征。
- **时序交叉验证**：避免未来信息泄露，使用 `TimeSeriesSplit`。

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

# 时序交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 参数网格
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05]
}

# 优化买入价模型
grid_buy = GridSearchCV(model_buy, param_grid, cv=tscv)
grid_buy.fit(X_train, y_buy_train)
print("Best Params for Buy Model:", grid_buy.best_params_)
```

---

### **6. 最终预测与部署**
保存模型并对新数据预测：

```python
import joblib

# 保存模型
joblib.dump(model_buy, 'model_buy.pkl')
joblib.dump(model_sell, 'model_sell.pkl')

# 加载模型并预测新数据
new_features = build_features(new_5min_data)
buy_price_pred = model_buy.predict(new_features)
sell_price_pred = model_sell.predict(new_features)
```

---

### **关键点总结**
1. **标签生成**：通过逐日计算成交量累计分位价格，确保目标定义清晰。
2. **特征工程**：聚合历史K线数据生成统计特征，捕捉价格和成交量的模式。
3. **模型训练**：使用LightGBM回归模型预测分位价格，优化超参数和验证方法。
4. **避免未来泄露**：在特征构建和交叉验证时严格遵循时序数据规则。

通过以上步骤，你可以高效预测每日的成交量驱动价格阈值，为交易策略提供参考。