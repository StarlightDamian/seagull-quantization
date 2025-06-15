# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:24:51 2024

@author: awei
训练lightGBM分类模型(train_1_lightgbm_classification)
"""
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb

from __init__ import path
from train import train_0_lightgbm

        
class LightgbmClassificationTrain(train_0_lightgbm.LightgbmTrain):
    def __init__(self, TARGET_REAL_NAMES=None):
        super().__init__(TARGET_REAL_NAMES)
        
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'binary',  # 二分类
            'num_leaves': 12,
            'learning_rate': 0.1,
            'metric': ['binary_logloss'],  # 二分类问题通常使用 binary_logloss
            'verbose': -1,  # 控制输出信息的详细程度，-1 表示不输出任何信息
        }
        lgb_classifier = lgb.LGBMClassifier(**params)
        self.model = MultiOutputClassifier(lgb_classifier)
        
        #train_model
        self.problem_type = 'classification'
