要将你提供的 **多标签二分类** 修改为 **多类分类**，我们需要做以下几个方面的调整：

1. **深度学习模型（MMLP）部分**：
   - 目前你使用的是 `BCEWithLogitsLoss`，这通常用于多标签二分类任务。对于多类分类任务，我们需要使用 **`CrossEntropyLoss`**，它适用于每个样本属于一个类别的情况。
   - **分类输出**应该是 **softmax** 激活函数，而不是 **sigmoid**，以确保输出值可以表示为每个类别的概率。
2. **标签处理**：
   - **`y_class_train`** 需要是 **一维数组**，其中每个样本有一个类别标签，标签范围为 `0, 1, 2, 3, 4`（对于 5 类分类任务）。当前你是 **多标签二分类**，你需要将标签格式更改为 **单标签分类**。
3. **LightGBM 部分**：
   - 对于 **多类分类**，你需要使用 `objective='multiclass'` 和 `num_class=5` 来指定类别数量。

### **修改后的代码**

以下是修改后的代码，适应 **多分类任务**：

#### **1. 修改深度学习部分（MMLP）**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_labels):
        super(MMLPModel, self).__init__()

        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.act1 = nn.SiLU()  # Swish activation

        # 隐藏层 1 到 2
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.act2 = nn.SiLU()

        # 隐藏层 2 到 3
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.act3 = nn.SiLU()

        # 解码器部分（重建原始输入）
        self.fc_dec = nn.Linear(hidden_units[2], input_dim)

        # 多分类任务头（5 个类别）
        self.classification_head = nn.Linear(hidden_units[2], num_labels)  # 多分类输出
        self.dropout = nn.Dropout(dropout_rates[0])

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # 解码器部分（如果需要重建原始输入）
        decoder_output = self.fc_dec(x)

        # 分类任务输出（softmax 激活用于多分类）
        classification_output = self.classification_head(x)  # 不加激活

        return decoder_output, classification_output

# 模型实例化
input_dim = 130  # 输入特征维度
hidden_units = [128, 64, 32]  # 隐藏层单元
dropout_rates = [0.2]  # dropout比率
num_labels = 5  # 分类标签数量

model = MMLPModel(input_dim, hidden_units, dropout_rates, num_labels)
```

#### **2. 训练 MMLP 模型（使用 CrossEntropyLoss）**

```python
from torch.utils.data import DataLoader, TensorDataset

# 假设训练数据
x_train = np.random.rand(1000, 130)  # 1000 个样本，130 个特征
y_class_train = np.random.choice([0, 1, 2, 3, 4], size=1000)  # 每个样本 1 个标签，类别为 [0, 1, 2, 3, 4]

# 转换为 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.long)  # 使用 long 类型作为标签

# 创建 DataLoader
train_dataset = TensorDataset(x_train_tensor, y_class_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 损失函数和优化器
criterion_class = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss_class = 0.0
    for i, (x_batch, y_class_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        decoder_output, class_output = model(x_batch)

        # 计算损失
        loss_class = criterion_class(class_output, y_class_batch)

        # 反向传播
        loss_class.backward()
        optimizer.step()

        running_loss_class += loss_class.item()

    print(f"Epoch {epoch+1}/{epochs}, Classification Loss: {running_loss_class / len(train_loader):.4f}")
```

#### **3. 训练 LightGBM 模型（使用多类分类目标）**

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 假设你已经获得了隐层编码作为特征，y_class_train 是标签
# 将多分类标签（0, 1, 2, 3, 4）作为目标
X_train = np.random.rand(1000, 128)  # 假设通过 MMLP 提取的隐层编码（128维）
y_class_train = np.random.choice([0, 1, 2, 3, 4], size=1000)  # 每个样本 1 个标签（类别0,1,2,3,4）

# 拆分数据集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_class_train, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# LightGBM 参数
params = {
    'objective': 'multiclass',  # 多类分类
    'num_class': 5,  # 5 个类别
    'metric': 'multi_logloss',  # 多类损失
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

# 训练模型
lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_val])

# 预测
y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

# 计算 ROC AUC 分数
auc_score = roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')
print("ROC AUC Score:", auc_score)
```

### **解释：**

1. **深度学习模型（MMLP）**：
   - 使用 **`CrossEntropyLoss`** 作为损失函数来处理 **多类分类**。
   - **`nn.CrossEntropyLoss()`** 适用于 **多分类** 问题，输出层不需要 `softmax`，因为 `CrossEntropyLoss` 会自动处理 `logits`（即未经过激活的输出）。
2. **LightGBM 模型**：
   - 对于 **多类分类**，使用 **`objective='multiclass'`** 和 **`num_class=5`** 来指定类别数（5 个类别）。
   - 使用 **`multi_logloss`** 作为评估指标，适用于多类分类问题。
3. **训练过程**：
   - 深度学习模型（MMLP）用于从输入特征中提取隐层表示。
   - 通过 `train_test_split` 将提取的隐层表示用于训练 LightGBM 模型。
   - 计算 **ROC AUC** 分数时，我们使用 `roc_auc_score` 来评估模型的多分类性能。

### **总结**：

- 你将 **多类分类** 任务转化为 **LightGBM** 训练任务。
- 深度学习部分的 **MMLP** 提供了一个强大的特征提取器，并且通过 **多类交叉熵损失** 来进行训练。
- 随后，使用 **LightGBM** 在提取的特征上进行多类分类任务。

如果你有更多问题或需要进一步调整模型，请随时告诉我！