# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:16:27 2024

@author: awei
(demo_train_2_softplus)
"""
import pandas as pd
import numpy as np
#from scipy.special import expit
#from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
import lightgbm as lgb  

from seagull.settings import PATH


# 自定义损失函数  
def custom_loss(y_pred, dataset):
    y_true = dataset.get_label()
    
    beta = 10.86  # 1.29#4.3#
    diff = y_true - y_pred  
    sigmoid = 1 / (1 + np.exp(beta * diff))
    #grad = diff * sigmoid
    #hess = beta * sigmoid * (1 - sigmoid) * diff  
    grad =  - beta * diff  * sigmoid  # 引导 y_pred 接近 y_true
    hess = sigmoid * (1 - sigmoid) * diff  #* beta  # 去除 diff 的影响
    #hess = np.ones_like(diff) * beta # 恒定的 Hessian，保证稳定性
    return grad, hess

# 自定义评估指标（例如，均方误差的变种）  
def custom_metric(y_pred, dataset):  
    y_true = dataset.get_label()
    diff = y_true - y_pred
    loss = diff * 1 / (1 + np.exp(-10 * diff))  # 使用与损失函数相似的形式，但只计算损失值  
    return 'custom_metric', np.mean(loss)  
  
# =============================================================================
# # 假设X_train_1, y_train, X_test_1, y_test已经定义并准备好了  
# params = {  
#     'objective': 'custom',  
#     'metric': 'custom',  
#     'verbose': -1,  
#     'learning_rate': 0.05,  
#     'num_leaves': 31,  
#     'min_data_in_leaf': 20,  
#     'max_depth': -1,  
#     'custom_objective': custom_loss,  # 注意：某些版本的LightGBM可能需要这种方式来指定自定义目标  
#     'custom_metric': custom_metric,   # 注意：某些版本的LightGBM可能需要这种方式来指定自定义评估指标  
#     # 或者，如果不支持上面的方式，您可以在lgb.train()中通过feval和feval_name参数指定  
# }  
# =============================================================================

# =============================================================================
# class CustomLoss(nn.Module):
#     def __init__(self, beta=10):
#         super(CustomLoss, self).__init__()
#         self.beta = beta
# 
#     def forward(self, y_pred, y_true):
#         diff = y_true - y_pred
#         loss = diff * torch.sigmoid(self.beta * diff)
#         return loss.mean()
#     
# def asw_objective(y_pred, dtrain, beta=10):
#     """
#     Adaptive Sigmoid-Weighted Loss objective function for LightGBM
#     
#     Args:
#         y_pred: array-like of shape (n_samples,)
#             The predicted values
#         dtrain: lgb.Dataset
#             Contains the ground truth labels
#         beta: float, default=10
#             Sigmoid steepness parameter
#             
#     Returns:
#         grad: First order gradients
#         hess: Second order gradients
#     """
#     y_true = dtrain.get_label()
#     diff = y_true - y_pred
#     
#     # Calculate sigmoid
#     sigmoid = expit(beta * diff)
#     
#     # Gradient: d(loss)/d(y_pred)
#     grad = -(sigmoid + beta * diff * sigmoid * (1 - sigmoid))
#     
#     # Hessian: d²(loss)/d(y_pred)²
#     hess = beta * sigmoid * (1 - sigmoid) * (1 + beta * diff * (1 - 2 * sigmoid))
#     
#     return grad, hess
# =============================================================================

# 自定义评估指标函数
def custom_metric(y_pred, dataset):
    y_true = dataset.get_label()
    data_type=dataset.params.get('data_type', '')
    if data_type == 'train':
        # 处理训练集的逻辑
        next_close_real = X_train.next_close_real
    elif data_type == 'valid':
        # 处理验证集的逻辑
        next_close_real = X_test.next_close_real
    
    # reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
    # total_profit = np.prod(reward, axis=0)**0.5
    # total_profit = np.mean(reward) 
    bias_array = np.arange(-0.5, 0.51, 0.003)
    y_pred_matrix = y_pred[:, np.newaxis] + bias_array
    
# =============================================================================
#     if data_type == 'train':
#         train_results = column_mean
#     elif data_type == 'valid':
#         valid_results = column_mean
# =============================================================================
    reward_matrix = np.where(y_pred_matrix <= y_true[:, np.newaxis],
                           y_pred_matrix - 1,
                           np.tile((next_close_real - 1).values[:, np.newaxis], (1, len(bias_array)))
                           )    

    reward_mean = np.mean(reward_matrix, axis=0)
    max_index = np.argmax(reward_mean, axis=0)
    print(f'{data_type}: {max_index}| {bias_array[max_index]:.5f}')
    global best_train_bias
    if data_type == 'train':
        best_train_bias = bias_array[max_index]
    total_profit = reward_mean[max_index]
    
    # 计算其他指标
    valid_preds = np.sum(y_pred <= y_true)
    total_preds = len(y_pred)
    valid_ratio = valid_preds / total_preds
    #print(f"Valid predictions ratio: {valid_ratio:.4f} ({valid_preds}/{total_preds})")

    return '|', total_profit, True  # 名称, 值, 是否越大越好
# =============================================================================
# def asw_metric(y_pred, dtrain, beta=10):
#     """
#     Adaptive Sigmoid-Weighted Loss metric for evaluation
#     
#     Args:
#         y_pred: array-like of shape (n_samples,)
#             The predicted values
#         dtrain: lgb.Dataset
#             Contains the ground truth labels
#         beta: float, default=10
#             Sigmoid steepness parameter
#             
#     Returns:
#         name: str
#             Name of the metric
#         score: float
#             Computed metric value
#         is_higher_better: bool
#             Whether a higher score is better
#     """
#     y_true = dtrain.get_label()
#     diff = y_true - y_pred
#     loss = diff * expit(beta * diff)
#     return 'asw_loss', float(np.mean(loss)), False
# =============================================================================

# 修改后的参数配置
params = {
    'objective': custom_loss,#lambda y_pred, dtrain: asw_objective(y_pred, dtrain, beta=10),
    'metric': 'custom',  # 使用自定义评估指标
    'verbose': -1,
    'learning_rate': 0.005,
    'num_leaves': 55,
    'min_data_in_leaf': 20,
    'max_depth': 10,
}

data = pd.read_csv(f'{PATH}/_file/test_603893.csv')
#data['high'] = data['high'] / data['close']
columns_to_divide = ['high', 'low', 'open', 'close']
data[columns_to_divide] = data[columns_to_divide].div(data['preclose'], axis=0)

data[['next_high', 'next_low','next_open']] = data[['high', 'low','open']].shift(-1)
data['next_close_real'] = data['close'].shift(-1)
data = data.head(-1)
feature_name=['open', 'high', 'low', 'close', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
       'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST','next_open','next_close_real']
X = data[feature_name]
#y = data[['next_high', 'next_low']]
y = data['next_high']
# 拆分数据集
# 假设 X 和 y 已经准备好
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

feature_name_1=['open', 'high', 'low', 'close', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
       'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']
X_train_1 = X_train[feature_name_1]
X_test_1 = X_test[feature_name_1]

# 创建训练和验证数据集
lgb_train = lgb.Dataset(
    X_train_1,
    label=y_train,
    free_raw_data=False,
    params={'data_type': 'train'}
)

lgb_valid = lgb.Dataset(
    X_test_1,
    label=y_test,
    reference=lgb_train,
    free_raw_data=False,
    params={'data_type': 'valid'}
)

# 训练模型
model = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    feval=custom_metric,#lambda y_pred, dtrain: asw_metric(y_pred, dtrain, beta=10),
    num_boost_round=100,
    callbacks=[lgb.callback.log_evaluation(period=10)]
)

# 预测
y_pred = model.predict(X_test_1)

result = pd.DataFrame([y_test.values, y_pred]).T
result.columns = ['y_test','y_pred']
print(result)
result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)

result.to_csv(f'{PATH}/_file/test_result_reward2.csv',index=False)

result_bool = result[result.next_high_bool==1]
y_test,y_pred,next_high_bool = result_bool.mean()
print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])

reward = (y_pred-1)*(result_bool.shape[0])
reward_all = (y_test-1)*(result.shape[0])
reward_pct = reward/reward_all
print(f'reward_pct: {reward_pct:.4f}')
# 计算最终的损失值
#final_loss = asw_metric(y_pred, lgb_valid)[1]
#print(f"Final ASW Loss: {final_loss:.4f}")

# =============================================================================
# import numpy as np
# from sklearn.metrics import make_scorer
# from scipy.special import expit  # sigmoid function
# 
# def adaptive_sigmoid_weighted_loss(y_true, y_pred, beta=10):
#     """
#     Adaptive Sigmoid-Weighted Loss (ASW Loss)
#     
#     A novel asymmetric loss function that adaptively weights prediction errors using 
#     a sigmoid function. The loss is particularly sensitive to underestimation or
#     overestimation based on the beta parameter.
#     
#     Parameters:
#     -----------
#     y_true : array-like of shape (n_samples,)
#         Ground truth target values.
#     y_pred : array-like of shape (n_samples,)
#         Predicted target values.
#     beta : float, default=10
#         Sigmoid steepness parameter. Higher values create sharper asymmetry.
#         
#     Returns:
#     --------
#     float
#         Computed loss value.
#     
#     References:
#     -----------
#     Similar to:
#     - Asymmetric Loss Functions (Mastouri et al., 2021)
#     - Adaptive Weight Learning (Chen et al., 2019)
#     """
#     diff = y_true - y_pred
#     loss = diff * expit(beta * diff)
#     return np.mean(loss)
# 
# def adaptive_sigmoid_weighted_gradient(y_true, y_pred, beta=10):
#     """
#     Gradient of the Adaptive Sigmoid-Weighted Loss function
#     
#     Parameters:
#     -----------
#     y_true : array-like of shape (n_samples,)
#         Ground truth target values.
#     y_pred : array-like of shape (n_samples,)
#         Predicted target values.
#     beta : float, default=10
#         Sigmoid steepness parameter.
#         
#     Returns:
#     --------
#     array-like of shape (n_samples,)
#         Gradient values for each sample.
#     """
#     diff = y_true - y_pred
#     sigmoid = expit(beta * diff)
#     # Derivative of loss with respect to y_pred
#     grad = -sigmoid - beta * diff * sigmoid * (1 - sigmoid)
#     return grad
# 
# # Create scorer for sklearn
# asw_scorer = make_scorer(adaptive_sigmoid_weighted_loss, 
#                         greater_is_better=False, 
#                         needs_proba=False)
# 
# # Example usage with sklearn:
# """
# from sklearn.linear_model import SGDRegressor
# 
# # Initialize model with custom gradient
# model = SGDRegressor(loss='modified_huber', # closest built-in loss
#                     learning_rate='constant',
#                     eta0=0.01,
#                     max_iter=1000)
# 
# # Custom training loop
# for epoch in range(n_epochs):
#     # Get predictions
#     y_pred = model.predict(X)
#     
#     # Calculate gradients using our custom gradient function
#     grads = adaptive_sigmoid_weighted_gradient(y_true, y_pred)
#     
#     # Update model parameters
#     model.coef_ -= learning_rate * grads.dot(X)
#     model.intercept_ -= learning_rate * grads.mean()
# """
# =============================================================================
# =============================================================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# 
# class CustomLoss(nn.Module):
#     def __init__(self, beta=10):
#         super(CustomLoss, self).__init__()
#         self.beta = beta
# 
#     def forward(self, y_pred, y_true):
#         diff = y_true - y_pred
#         loss = diff * torch.sigmoid(self.beta * diff)
#         return loss.mean()
# =============================================================================
