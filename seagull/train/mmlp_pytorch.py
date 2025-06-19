# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:34:10 2024

@author: awei
https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp?scriptVersionId=73762661&cellId=2
Jane Street mmlp
mmlp_pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_labels):
        super(MMLPModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.act1 = nn.SiLU()  # Swish activation

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.act2 = nn.SiLU()

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.act3 = nn.SiLU()

        self.fc_dec = nn.Linear(hidden_units[2], input_dim)  # Decoder output

        # 分类任务头
        self.classification_head = nn.Linear(hidden_units[2], num_labels)  # 分类任务
        # 回归任务头
        self.regression_head = nn.Linear(hidden_units[2], num_labels)  # 回归任务

        self.dropout = nn.Dropout(dropout_rates[0])

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # 解码器部分
        decoder_output = self.fc_dec(x)

        # 分类任务输出
        classification_output = torch.sigmoid(self.classification_head(x))

        # 回归任务输出
        regression_output = self.regression_head(x)

        return decoder_output, classification_output, regression_output, x  # 返回隐层编码 x

# 模型实例化
input_dim = 130  # 输入特征维度
hidden_units = [128, 64, 32]  # 隐藏层单元
dropout_rates = [0.2]  # dropout比率
num_labels = 5  # 分类标签数量

model = MMLPModel(input_dim, hidden_units, dropout_rates, num_labels)
# 假设你有 x_train 和 y_class_train
# 使用 PyTorch 训练 MMLP 模型

from torch.utils.data import DataLoader, TensorDataset

# 假设训练数据
x_train = np.random.rand(1000, 130)  # 1000 个样本，130 个特征
#y_class_train = np.random.randint(0, 2, (1000, 5))  # 5 个分类任务标签
y_class_train = np.random.choice([0, 1, 2, 3, 4], size=1000)

# 转换为 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(x_train_tensor, y_class_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 损失函数和优化器
criterion_class = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss_class = 0.0
    for i, (x_batch, y_class_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        decoder_output, class_output, reg_output, hidden_encoding = model(x_batch)

        # 计算损失
        loss_class = criterion_class(class_output.squeeze(), y_class_batch)

        # 反向传播
        loss_class.backward()
        optimizer.step()

        running_loss_class += loss_class.item()

    print(f"Epoch {epoch+1}/{epochs}, Classification Loss: {running_loss_class / len(train_loader):.4f}")

# 提取隐层编码（hidden_encoding）
model.eval()
with torch.no_grad():
    hidden_encoding = []
    for x_batch, _ in train_loader:
        _, _, _, encoding = model(x_batch)
        hidden_encoding.append(encoding)
    
    hidden_encoding = torch.cat(hidden_encoding, dim=0)
    hidden_encoding = hidden_encoding.numpy()  # 转换为 numpy 数组
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 假设你已经获得了隐层编码作为特征，y_class_train 是标签
X_train, X_val, y_train, y_val = train_test_split(hidden_encoding, y_class_train, test_size=0.2, random_state=42)

# 训练 LightGBM 模型
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

# 训练模型
lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_val])#, early_stopping_rounds=10

# 预测
y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

# 计算 ROC AUC 分数
auc_score = roc_auc_score(y_val, y_pred)
print("ROC AUC Score:", auc_score)
