要将你提供的 TensorFlow/Keras 代码修改为 **PyTorch** 版本，并确保它能够正常工作并进行模块化的测试，我们需要进行一些重要的调整。下面是如何将代码转换为 PyTorch 版本的步骤：

1. **模型结构转换**：TensorFlow/Keras 中的层需要转换为 PyTorch 中的层（例如 `nn.Linear`、`nn.BatchNorm1d`、`nn.ReLU` 等）。
2. **损失函数和优化器**：PyTorch 中使用的损失函数和优化器与 Keras 不同，我们将使用 PyTorch 中的对应函数（如 `nn.BCEWithLogitsLoss`）。
3. **训练和验证循环**：PyTorch 中的训练和验证过程需要手动实现，而不是通过 Keras 的 `fit()` 方法。
4. **数据加载**：使用 PyTorch 的 `DataLoader` 来处理批量数据加载和并行化。

### **转换后的 PyTorch 代码**

以下是基于你提供的 TensorFlow/Keras 模型，转换为 PyTorch 版本的代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 模拟数据加载（请替换为实际的数据加载代码）
# x_train: 特征数据, y_class_train: 分类标签, y_reg_train: 回归目标值
# 这里假设 train 是一个 DataFrame，其中包含特征、目标变量、日期等信息
x_train = np.random.rand(100, 130)  # 假设有 100 个样本和 130 个特征
y_class_train = np.random.randint(0, 2, (100, 5))  # 二分类任务的标签
y_reg_train = np.random.rand(100, 5)  # 回归任务的标签

# 转换为 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.float32)
y_reg_train_tensor = torch.tensor(y_reg_train, dtype=torch.float32)

# 使用 DataLoader 进行批量训练
train_dataset = TensorDataset(x_train_tensor, y_class_train_tensor, y_reg_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建多任务学习模型
class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_labels):
        super(MMLPModel, self).__init__()

        self.input_dim = input_dim

        # 编码器部分
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.act1 = nn.SiLU()  # Swish activation

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.act2 = nn.SiLU()

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.act3 = nn.SiLU()

        # 解码器部分
        self.fc_dec = nn.Linear(hidden_units[2], input_dim)  # Decoder output

        # 多任务输出
        self.classification_head = nn.Linear(hidden_units[2], num_labels)  # 分类任务
        self.regression_head = nn.Linear(hidden_units[2], num_labels)  # 回归任务

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # 解码器部分（重建原始输入）
        decoder_output = self.fc_dec(x)

        # 分类任务输出（sigmoid 激活）
        classification_output = torch.sigmoid(self.classification_head(x))

        # 回归任务输出
        regression_output = self.regression_head(x)

        return decoder_output, classification_output, regression_output

# 设置模型超参数
input_dim = x_train.shape[1]  # 输入特征数量
hidden_units = [128, 64, 32]  # 隐藏层大小
dropout_rates = [0.2, 0.2, 0.2]  # dropout比率
num_labels = 5  # 分类和回归任务的标签数

# 初始化模型
model = MMLPModel(input_dim, hidden_units, dropout_rates, num_labels)

# 损失函数
criterion_class = nn.BCEWithLogitsLoss()  # 二分类交叉熵
criterion_reg = nn.MSELoss()  # 均方误差损失

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
def train_model(model, train_loader, criterion_class, criterion_reg, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss_class = 0.0
        running_loss_reg = 0.0
        for i, (x_batch, y_class_batch, y_reg_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            decoder_output, class_output, reg_output = model(x_batch)

            # 计算损失
            loss_class = criterion_class(class_output.squeeze(), y_class_batch)
            loss_reg = criterion_reg(reg_output.squeeze(), y_reg_batch)

            # 总损失
            total_loss = loss_class + loss_reg
            total_loss.backward()

            optimizer.step()

            running_loss_class += loss_class.item()
            running_loss_reg += loss_reg.item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Classification Loss: {running_loss_class / len(train_loader):.4f}, "
              f"Regression Loss: {running_loss_reg / len(train_loader):.4f}")

# 训练模型
train_model(model, train_loader, criterion_class, criterion_reg, optimizer, epochs=5)

# 测试阶段（如果有测试数据，进行预测）
model.eval()  # 切换到评估模式
with torch.no_grad():
    # 假设你有测试数据 x_test
    x_test = np.random.rand(20, 130).astype(np.float32)
    x_test_tensor = torch.tensor(x_test)
    decoder_pred, class_pred, reg_pred = model(x_test_tensor)

    print("Classification Predictions:", class_pred)
    print("Regression Predictions:", reg_pred)
```

### **代码解析：**

1. **模型结构**：

   - **编码器部分**：通过几层 **全连接层（`Linear`）**、**批量归一化（`BatchNorm1d`）** 和 **激活函数（`SiLU`，即 Swish）** 来提取特征。

   - **解码器部分**：将编码器的输出映射回输入空间（重建输入数据），并通过一个输出层得到与输入数据相同维度的重建结果。

   - 任务头

     ：

     - **分类头**：输出 0 到 1 的概率值，用于二分类任务（`sigmoid` 激活）。
     - **回归头**：输出连续的数值，用于回归任务。

2. **损失函数**：

   - **分类任务损失**：使用 `BCEWithLogitsLoss`（带 `sigmoid` 的二元交叉熵）。
   - **回归任务损失**：使用 `MSELoss`（均方误差）。

3. **优化器**：

   - 使用 **Adam** 优化器进行训练。

4. **训练循环**：

   - 对每个批次进行前向传播、计算损失、反向传播并更新模型权重。

5. **评估与预测**：

   - 在训练之后，使用 `model.eval()` 切换到评估模式，进行预测时关闭 **dropout**，使用 **`torch.no_grad()`** 来禁用梯度计算。

### **总结**

- 这段代码实现了一个多任务学习（Multi-task Learning）框架，包含 **分类任务** 和 **回归任务**，并使用 PyTorch 来实现。
- 模型结构包括 **编码器和解码器**，通过 **全连接层**、**批量归一化** 和 **Swish 激活函数** 构建。
- 使用 **`BCEWithLogitsLoss`** 和 **`MSELoss`** 作为分类和回归任务的损失函数，优化器使用 **Adam**。
- 训练过程中，模型同时优化这两个任务的损失。

如果你有更多问题，或者希望对模型进行改进，请随时告诉我！