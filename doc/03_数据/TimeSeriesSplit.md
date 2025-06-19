`sklearn.model_selection.TimeSeriesSplit` 是一个非常适合时间序列问题的交叉验证方法。它保证在进行训练和验证时，始终遵循时间顺序，即训练数据仅包括历史数据，而验证数据则是未来数据。这里，我将提供一个示例代码，使用 `high`、`low`、`close` 作为特征，数据包含股票的主键、日期和股票代码。

### 示例代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 示例数据：主键、日期、股票代码、收盘价等
# 假设已经加载了一个包含这些列的数据框
data = {
    'stock_code': ['000001', '000001', '000001', '000001', '000001', '000002', '000002', '000002', '000002', '000002'],
    'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
    'high': [10.5, 11.0, 10.8, 10.9, 11.2, 20.5, 21.0, 20.8, 20.9, 21.2],
    'low': [10.0, 10.5, 10.3, 10.4, 10.7, 19.5, 20.0, 19.8, 19.9, 20.2],
    'close': [10.2, 10.8, 10.5, 10.7, 11.0, 20.2, 20.5, 20.3, 20.6, 20.8]
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# 假设我们要预测 'close' 作为目标变量，其它的 'high', 'low' 作为特征
X = df[['high', 'low']]  # 特征
y = df['close']  # 目标变量

# 股票代码作为一个额外的信息列
stocks = df['stock_code'].unique()

# 初始化 TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# 对每一个股票进行滚动交叉验证
for stock in stocks:
    # 过滤出当前股票的数据
    stock_data = df[df['stock_code'] == stock]
    
    # 定义 X 和 y
    X_stock = stock_data[['high', 'low']]
    y_stock = stock_data['close']
    
    print(f"\nStock Code: {stock}")
    
    # 进行滚动交叉验证
    for train_index, test_index in tscv.split(X_stock):
        X_train, X_test = X_stock.iloc[train_index], X_stock.iloc[test_index]
        y_train, y_test = y_stock.iloc[train_index], y_stock.iloc[test_index]
        
        # 创建并训练模型（这里使用简单的线性回归作为示例）
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测并计算均方误差
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Train Period: {stock_data['date'].iloc[train_index[0]]} - {stock_data['date'].iloc[train_index[-1]]}")
        print(f"Test Period: {stock_data['date'].iloc[test_index[0]]} - {stock_data['date'].iloc[test_index[-1]]}")
        print(f"Mean Squared Error: {mse:.4f}")
```

### 解释

1.  **数据准备：**
    -   `df` 是一个包含股票的历史数据的 DataFrame，包含了 `stock_code`（股票代码）、`date`（日期）、`high`（最高价）、`low`（最低价）、`close`（收盘价）等字段。
    -   我们选择 `high` 和 `low` 作为特征，`close` 作为目标变量。
2.  **时间序列划分：**
    -   使用 `TimeSeriesSplit(n_splits=3)`，将数据分成 3 个训练和验证集。这个方法会确保每次训练集都只包含历史数据，且验证集包含未来数据。
    -   交叉验证每次切分时，数据会从训练集向后滚动，避免了未来数据泄露问题。
3.  **按股票代码划分：**
    -   因为每个股票的历史数据独立性较强，我们对每个股票进行单独的时间序列交叉验证。
    -   `tscv.split()` 会返回训练集和测试集的索引，训练集总是包含早期的数据，测试集总是包含后期的数据。
4.  **模型训练与评估：**
    -   使用 `LinearRegression` 作为示例模型，对每个划分的训练集进行训练。
    -   使用 `mean_squared_error` 计算每个验证集的均方误差 (MSE)，你可以根据自己的需求更换为其他评估指标。

### 注意事项：

-   **数据的时间顺序：** 确保每个划分时的训练集和测试集时间是连续的，不会出现未来数据泄露。
-   **每个股票独立处理：** 这里的例子是按股票代码单独处理的，每个股票的数据都是独立的，您可以根据需要修改代码使其适用于全局数据集。
-   **股票数据的粒度：** 该示例是按日交易数据来划分的。如果数据粒度为分钟或其他时间间隔，可以根据实际情况调整数据处理和模型的时间窗口。

### 总结

这种方式保证了时间序列数据的严格时间顺序，防止了未来数据泄露，适用于金融市场预测等问题。通过 `TimeSeriesSplit`，可以实现时间递增的训练与验证集划分，有效避免了传统交叉验证方法中的数据泄漏问题。