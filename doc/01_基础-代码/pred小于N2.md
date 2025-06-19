非常抱歉之前理解有误。我现在完全理解您的需求了。您希望模型的预测结果满足以下条件：

1. 在大约86%的情况下，预测值应该小于实际值。
2. 在剩余约14%的情况下，预测值应该大于实际值。
3. 同时，整体预测误差应该尽可能小。

这确实是一个特殊的回归问题，我们可以通过自定义损失函数来实现这个目标。我们可以使用LightGBM的自定义目标函数功能来实现这一点。

下面我将为您提供一个示例代码，展示如何使用LightGBM实现这个特定需求：



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# 自定义目标函数
def custom_asymmetric_objective(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -0.86, 0.14)
    hess = np.ones_like(residual)
    return grad, hess

def custom_asymmetric_eval(y_true, y_pred):
    residual = y_true - y_pred
    loss = np.where(residual > 0, 
                    (residual ** 2) * 0.86, 
                    (residual ** 2) * 0.14)
    return "custom_asymmetric_eval", np.mean(loss), False

# 假设我们已经有了特征数据 X 和目标变量 y
# X = pd.DataFrame(...)  # 您的特征数据
# y = pd.Series(...)     # 实际的股价

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': custom_asymmetric_objective,
    'metric': 'custom',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 训练模型
model = lgb.train(params, train_data, num_boost_round=100, 
                  valid_sets=[test_data], 
                  feval=custom_asymmetric_eval,
                  early_stopping_rounds=10)

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
    below_actual = np.mean(y_pred < y_true)
    mse = np.mean((y_true - y_pred)**2)
    return below_actual, mse

below_actual, mse = custom_metric(y_test, y_pred)
print(f"Custom Metric - Below Actual: {below_actual:.2%}, MSE: {mse:.4f}")

```

这段代码实现了以下功能：

1. 定义了一个自定义目标函数 `custom_asymmetric_objective`，它对正残差（预测值小于实际值）和负残差（预测值大于实际值）给予不同的权重，分别是0.86和0.14。

2. 定义了一个自定义评估函数 `custom_asymmetric_eval`，用于在训练过程中评估模型性能。

3. 使用LightGBM训练模型，应用自定义目标函数和评估函数。

4. 评估模型性能，包括均方误差（MSE）和预测值低于实际值的比例。

5. 可视化预测结果和实际值的对比，以及预测误差的分布。

要使用这段代码，您需要：

1. 准备好您的特征数据（X）和目标变量（y）。
2. 根据您的数据特点，可能需要调整模型参数，如 `num_leaves`, `learning_rate`, `feature_fraction` 等。
3. 运行代码，观察输出的评估指标和图表。

如果预测值低于实际值的比例不够接近86%，您可以尝试调整自定义目标函数中的权重。如果比例太高，稍微降低0.86的权重；如果太低，稍微提高0.86的权重。

此外，您还可以考虑以下优化方向：

1. 特征工程：创建更有预测力的特征。
2. 超参数调优：使用网格搜索或贝叶斯优化来找到最佳的模型参数。
3. 调整训练轮数：增加或减少 `num_boost_round` 参数。

如果您需要针对特定步骤的更多细节或者有任何问题，请随时告诉我。我很乐意提供更多帮助！