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
y_reg_train = np.random.rand(100, 2)  # 回归任务的标签

# 转换为 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.float32)
y_reg_train_tensor = torch.tensor(y_reg_train, dtype=torch.float32)

# 使用 DataLoader 进行批量训练
train_dataset = TensorDataset(x_train_tensor, y_class_train_tensor, y_reg_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建多任务学习模型
class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_class_labels,num_reg_labels):
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
        self.classification_head = nn.Linear(hidden_units[2], num_class_labels)  # 分类任务
        self.regression_head = nn.Linear(hidden_units[2], num_reg_labels)  # 回归任务

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
num_class_labels = y_class_train.shape[1]#5  # 分类和回归任务的标签数
num_reg_labels= y_reg_train.shape[1]#2
# 初始化模型
model = MMLPModel(input_dim, hidden_units, dropout_rates, num_class_labels,num_reg_labels)

# 损失函数
criterion_class = nn.BCEWithLogitsLoss()  # 二分类交叉熵
criterion_reg = nn.MSELoss()  # 均方误差损失

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs=10
# 训练过程
#def train_model(model, train_loader, criterion_class, criterion_reg, optimizer, epochs=10):
for epoch in range(epochs):
    model.train()
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
#train_model(model, train_loader, criterion_class, criterion_reg, optimizer, epochs=5)

# 测试阶段（如果有测试数据，进行预测）
model.eval()  # 切换到评估模式
with torch.no_grad():
    # 假设你有测试数据 x_test
    x_test = np.random.rand(20, 130).astype(np.float32)
    x_test_tensor = torch.tensor(x_test)
    decoder_pred, class_pred, reg_pred = model(x_test_tensor)

    print("Classification Predictions:", class_pred)
    print("Regression Predictions:", reg_pred)
