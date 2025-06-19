理解您的需求。您希望使用LightGBM进行股价上限预测，同时满足以下条件：

1. 86%的预测价格应低于实际价格
2. 在满足上述条件的同时，预测值应尽可能接近实际价格

这个任务实际上是一个分位数回归问题，我们需要预测股价的0.86分位数。我们可以使用LightGBM的分位数回归功能来实现这一目标。下面我'll为您提供一个示例代码，展示如何使用sklearn API的LightGBM来完成这个任务。



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# 假设我们已经有了特征数据 X 和目标变量 y
# X = pd.DataFrame(...)  # 您的特征数据
# y = pd.Series(...)     # 实际的股价上限

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LGBMRegressor(
    objective='quantile',
    alpha=0.86,  # 设置为0.86分位数
    n_estimators=100,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 计算预测值低于实际值的比例
below_actual = np.mean(y_pred < y_test)
print(f"Proportion of predictions below actual: {below_actual:.2%}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

# 分析预测误差
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# 自定义评估函数
def custom_metric(y_true, y_pred):
    errors = y_true - y_pred
    below_actual = np.mean(y_pred < y_true)
    mse = np.mean(errors**2)
    return below_actual, mse

below_actual, mse = custom_metric(y_test, y_pred)
print(f"Custom Metric - Below Actual: {below_actual:.2%}, MSE: {mse:.4f}")

```

这段代码实现了以下功能：

1. 使用LightGBM进行分位数回归，预测0.86分位数的股价。
2. 评估模型性能，包括均方误差（MSE）和预测值低于实际值的比例。
3. 可视化预测结果和实际值的对比。
4. 分析预测误差的分布。
5. 提供了一个自定义评估函数，同时考虑预测值低于实际值的比例和均方误差。

要使用这段代码，您需要：

1. 准备好您的特征数据（X）和目标变量（y）。
2. 根据您的数据特点，可能需要调整模型参数，如 `n_estimators`, `max_depth`, `learning_rate` 等。
3. 运行代码，观察输出的评估指标和图表。

如果预测值低于实际值的比例不够接近86%，您可以尝试调整 `alpha` 参数。如果比例太高，稍微降低 `alpha`；如果太低，稍微提高 `alpha`。

此外，您还可以考虑以下优化方向：

1. 特征工程：创建更有预测力的特征。
2. 超参数调优：使用网格搜索或贝叶斯优化来找到最佳的模型参数。
3. 集成方法：尝试将多个模型的预测结果合并。

如果您需要针对特定步骤的更多细节或者有任何问题，请随时告诉我。我很乐意提供更多帮助！