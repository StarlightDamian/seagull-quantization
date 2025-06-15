# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
训练lightGBM回归模型(train_1_lightgbm_regression)
"""
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from __init__ import path
from train import train_0_lightgbm


class LightgbmRegressionTrain(train_0_lightgbm.LightgbmTrain):
    def __init__(self, TARGET_REAL_NAMES=None):
        super().__init__(TARGET_REAL_NAMES)
        
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 12, # 决策树上的叶子节点的数量，控制树的复杂度
            'learning_rate': 0.1,  # 0.05,
            'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
            #w×RMSE+(1−w)×MAE
            'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
        }
        # loading data
        lgb_regressor = lgb.LGBMRegressor(**params)
        self.model = MultiOutputRegressor(lgb_regressor)
        
        # train_model
        self.problem_type = 'regression'
        
        
        
        