# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:35:28 2024

@author: awei
(softplus)
"""
import torch  
import torch.nn as nn 
import matplotlib.pyplot as plt  
class CustomLoss(nn.Module):
    def __init__(self, beta=10):
        super(CustomLoss, self).__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = diff * torch.sigmoid(self.beta * diff)
        return loss.mean()

if __name__ == '__main__': 
    y_preds = torch.linspace(0.97, 1.10, 100)
    y_true = torch.tensor([1.05])
    # 实例化自定义损失函数，并计算损失
    custom_loss = CustomLoss(beta=200)
    #loss = custom_loss(y_preds, y_true)  
    losses = []  

    # 对于每个 y_pred 值，计算损失  
    for y_pred in y_preds:  
        y_pred_tensor = y_pred.view(1)  # 确保 y_pred 是正确的形状以匹配 y_true  
        loss = custom_loss(y_pred_tensor, y_true)
        losses.append(loss.item())  
       
    print("Loss:", loss.item())
    
    plt.plot(y_preds.numpy(), losses)  
    plt.xlabel('y_pred')  
    plt.ylabel('Loss')  
    plt.title('Custom Loss Function Curve')  
    plt.grid(True)  
    plt.show()
# =============================================================================
# 
# import torch  
# import matplotlib.pyplot as plt  
# import numpy as np
# # 定义损失函数（与CustomLoss类中的forward方法相同）  
# def custom_loss_function(beta, y_pred, y_true):  
#     diff = torch.tensor(y_true - y_pred)
#     #loss =   
#     return diff * torch.sigmoid(beta * diff)
#   
# # 设置beta值  
# beta = 10.0  
#   
# # 创建一个差值diff的范围  
# #diffs = torch.linspace(-2, 2, 400)  # 从-2到2，共400个点  
# y_pred = np.random.uniform(1.05, 1.08, 100)
# y_true = np.random.uniform(1.06, 1.09, 100)
# # 计算每个diff对应的损失值  
# losses = custom_loss_function(beta, y_pred, y_true)  
# diffs_numpy = y_true - y_pred
# # 绘制损失曲线  
# plt.plot(diffs_numpy, losses.numpy())  
# plt.xlabel('Difference (y_true - y_pred)')  
# plt.ylabel('Loss')  
# plt.title('Custom Loss Function Curve')  
# plt.grid(True)  
# plt.show()
# =============================================================================
