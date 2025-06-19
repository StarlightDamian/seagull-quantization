# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:28:15 2024

@author: awei
(demo_train_batch_reward)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import lightgbm as lgb

from seagull.settings import PATH


class BatchRewardTrainer:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
    
    def custom_loss_1(self):
        next_close_real = X_train.next_close_real
        reward = np.where(y_pred <= y_true, y_pred - 1, next_close_real - 1)
    def custom_loss(self, preds, train_data):
        """自定义批次reward计算的目标函数"""
        labels = train_data.get_label()
        
        # 重塑预测值和标签为批次形式
        n_samples = len(preds)
        n_complete_batches = n_samples // self.batch_size
        
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        
        # 按批次计算梯度和海森矩阵
        for i in range(n_complete_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            
            batch_preds = preds[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # 计算批次reward（示例：使用批次平均误差的负值作为reward）
            batch_reward = -np.mean(np.abs(batch_preds - batch_labels))
            
            # 计算梯度（简化版本）
            batch_grad = -2 * (batch_labels - batch_preds) / self.batch_size
            grad[start_idx:end_idx] = batch_grad
            
            # 计算海森矩阵（简化版本）
            batch_hess = np.ones_like(batch_preds) * 2 / self.batch_size
            hess[start_idx:end_idx] = batch_hess
        
        # 处理最后一个不完整的批次
        if n_samples % self.batch_size != 0:
            start_idx = n_complete_batches * self.batch_size
            batch_preds = preds[start_idx:]
            batch_labels = labels[start_idx:]
            
            batch_reward = -np.mean(np.abs(batch_preds - batch_labels))
            batch_grad = -2 * (batch_labels - batch_preds) / len(batch_preds)
            batch_hess = np.ones_like(batch_preds) * 2 / len(batch_preds)
            
            grad[start_idx:] = batch_grad
            hess[start_idx:] = batch_hess
            
        return grad, hess

    def train(self, X, y, test_size=0.2, random_state=42):
        """训练模型"""
        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
# =============================================================================
#         # 添加批次序号
#         batch_size = 32
#         num_rows = len(X_train)
#         batch_ids = (np.arange(num_rows) // batch_size).astype(int)
#         X_train['batch_id'] = batch_ids  
# =============================================================================
        
        feature_name_1=['open', 'high', 'low', 'close', 'volume',
               'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
               'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']
        X_train = X_train[feature_name_1]
        X_test = X_test[feature_name_1]
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 设置参数
        params = {
            'objective': self.custom_loss,#'custom_loss',#'none',  # 使用自定义目标函数
            'metric': 'None',     # 不使用内置评估指标
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': self.batch_size,  # 确保每个叶节点至少有一个完整批次
            'verbose': -1
        }
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, valid_data],
            #feval=custom_metric,#fobj=self.batch_reward_objective,
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        return model

if __name__ == '__main__': 
    # 生成示例数据
    #np.random.seed(42)
    #X = np.random.rand(1000, 10)
    #y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)
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
    # 创建训练器实例
    trainer = BatchRewardTrainer(batch_size=32)
    
    # 训练模型
    model = trainer.train(X, y)
    
# =============================================================================
#     feature_name_1=['open', 'high', 'low', 'close', 'volume',
#            'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
#            'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']
#     X_train_1 = X_train[feature_name_1]
#     X_test_1 = X_test[feature_name_1]
# =============================================================================

# =============================================================================
#     y_pred = model.predict(X_test)
#     result = pd.DataFrame([y_test.values, y_pred]).T
#     result.columns = ['y_test','y_pred']
#     print(result)
#     result['next_high_bool'] = np.where(result['y_test'] >= result['y_pred'], 1, None)
# 
#     result.to_csv(f'{PATH}/_file/test_result_reward2.csv',index=False)
# 
#     result_bool = result[result.next_high_bool==1]
#     y_test,y_pred,next_high_bool = result_bool.mean()
#     print(result_bool.mean(), result_bool.shape[0],'/',result.shape[0])
# 
#     reward = (y_pred-1)*(result_bool.shape[0])
#     reward_all = (y_test-1)*(result.shape[0])
#     reward_pct = reward/reward_all
#     print(f'reward_pct: {reward_pct:.4f}')
# =============================================================================
