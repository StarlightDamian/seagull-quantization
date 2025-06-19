# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:12:04 2024

@author: awei
lightgbm_base
"""
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error
import lightgbm as lgb
import joblib
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, make_scorer

from seagull.settings import PATH
from seagull.utils import utils_database, utils_character, utils_log, utils_math
from rolling_cv import RollingCV
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')



def f05_score(y_true, y_pred):
    """
    计算 F0.5 score
    F0.5 = (1 + 0.5^2) * (precision * recall) / (0.5^2 * precision + recall)
    """
    return fbeta_score(y_true, y_pred, beta=0.5, average='weighted')
f05_scorer = make_scorer(f05_score)

class StockModelPipeline:
    def __init__(self, n_clusters=15, random_state=42):
        self.random_state = random_state

        # 初始化模型
        self.class_pipeline = None
        self.reg_pipeline = None
        self.grid_search = None
    def preprocess_data(self, X, numeric_features, categorical_features):
        """
        构建特征预处理流水线
        """
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
# =============================================================================
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
#                 ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder())]), categorical_features)
#             ]
#         )
# =============================================================================
        return preprocessor

    def fit_cluster(self, x_train, numeric_features, categorical_features):
        """
        训练聚类模型
        """
        preprocessor = self.preprocess_data(x_train, numeric_features, categorical_features)
        self.cluster_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('kmeans', KMeans(n_clusters=self.n_clusters, random_state=self.random_state))
        ])
        self.cluster_pipeline.fit(x_train)
        cluster_labels = self.cluster_pipeline.named_steps['kmeans'].labels_
        return cluster_labels

    def fit_class_model(self, x_train, y_train, numeric_features, categorical_features, param_grid=None):
        """
        训练分类模型，并进行超参数优化
        """
        preprocessor = self.preprocess_data(x_train, numeric_features, categorical_features)
        classifier = lgb.LGBMClassifier(random_state=self.random_state,
                                        force_col_wise=True,
                                        #logging_level='error',  # 只输出错误信息
                                        verbose=-1,
                                       # importance_type='split',#'split', # 会降低评估指标
                                        )  # 通过GOSS来加速训练, boosting_type='goss',
    
        # 创建pipeline
        self.class_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
        # 如果有超参数网格，使用GridSearchCV进行调参
        if param_grid:
            #GridSearchCV
            #Definition : RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch="2*n_jobs", random_state=None, error_score=np.nan, return_train_score=False)

            rolling_cv = RollingCV(n_splits=3, train_days=180, gap_days=2, val_rate=0.2)

            grid_search = RandomizedSearchCV(self.class_pipeline,
                                       param_grid,
                                       n_iter=10,  # 进行10次随机搜索
                                       cv=rolling_cv,
                                       scoring=f05_scorer,# 使用 F0.5 score 作为评估标准
                                       verbose=0, 
                                       n_jobs=-1,# 使用所有核心进行并行训练
                                       )  
            grid_search.fit(x_train, y_train)  # 传递样本权重
            
            # 获取最佳模型
            self.class_pipeline = grid_search.best_estimator_
            self.grid_search = grid_search
            
        else:
            # 没有网格时，直接训练并传入样本权重
            # 当数据集的特征数很多，或者特征稀疏性很高时，force_col_wise=True 可能会显著提高训练速度和减少内存使用。
            self.class_pipeline.fit(x_train, y_train, early_stopping_rounds=100, force_col_wise=True)
            
    def fit_reg_model(self, x_train, y_train, numeric_features, categorical_features, param_grid=None, regressor=None):
        """
        训练回归模型，并进行超参数优化
        """
        # 创建预处理器
        preprocessor = self.preprocess_data(x_train, numeric_features, categorical_features)
        
        # 如果未提供自定义回归器，则使用默认的 LGBMRegressor
        if regressor is None:
            regressor = lgb.LGBMRegressor(random_state=self.random_state)
        
        # 创建 pipeline
        self.reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
    
        # 执行网格搜索以优化超参数
        if param_grid:
            grid_search = GridSearchCV(self.reg_pipeline,
                                       param_grid,
                                       cv=3,
                                       scoring='neg_mean_squared_error')
            grid_search.fit(x_train, y_train)
            self.reg_pipeline = grid_search.best_estimator_
        else:
            # 使用 set_params 传递 early_stopping_rounds 给 regressor
            #self.reg_pipeline.set_params(regressor__early_stopping_rounds=100)
    
            # 训练模型
            self.reg_pipeline.fit(x_train, y_train)
    
        # 保存回归器对象
        self.regressor_ = self.reg_pipeline.named_steps['regressor']


# =============================================================================
#     def fit_reg_model(self, x_train, y_train, numeric_features, categorical_features, param_grid=None):
#         """
#         训练回归模型，并进行超参数优化
#         """
#         preprocessor = self.preprocess_data(x_train, numeric_features, categorical_features)
#         regressor = lgb.LGBMRegressor(random_state=self.random_state)
#         self.reg_pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('regressor', regressor)
#         ])
# 
#         if param_grid:
#             grid_search = GridSearchCV(self.reg_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
#             grid_search.fit(x_train, y_train)
#             self.reg_pipeline = grid_search.best_estimator_
#         else:
#             self.reg_pipeline.fit(x_train, y_train)
# =============================================================================

    def predict_class(self, x_test):
        """
        使用分类模型预测
        """
        return self.class_pipeline.predict(x_test)

    def predict_reg(self, x_test):
        """
        使用回归模型预测
        """
        return self.reg_pipeline.predict(x_test)

    def save_models(self, cluster_path, class_path, reg_path):
        """
        保存模型
        """
        joblib.dump(self.cluster_pipeline, cluster_path)
        joblib.dump(self.class_pipeline, class_path)
        joblib.dump(self.reg_pipeline, reg_path)

    def load_models(self, cluster_path, class_path, reg_path):
        """
        加载模型
        """
        self.cluster_pipeline = joblib.load(cluster_path)
        self.class_pipeline = joblib.load(class_path)
        self.reg_pipeline = joblib.load(reg_path)

    def extract_cluster_labels(self, x):
        """
        提取聚类标签
        """
        return self.cluster_pipeline.named_steps['kmeans'].predict(
            self.cluster_pipeline.named_steps['preprocessor'].transform(x)
        )
    
