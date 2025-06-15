# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:53:35 2024

@author: awei
lightGBM模型训练(train_0_lightgbm)
target_pred_names
"""
import os
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from __init__ import path
from feature import feature_engineering_main
from utils import utils_database, utils_character, utils_log, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

TEST_SIZE = 0.15 #数据集分割中测试集占比
TRAIN_MODEL_TABLE_NAME = 'ads_info_incr_train_model'#'train_model'

class LightgbmTrain(feature_engineering_main.FeatureEngineering):
    def __init__(self, target_names=None):
        self.multioutput_model_path = None
        self.target_names = target_names
        self.model = None
        self.test_size = TEST_SIZE
        
        # train_model
        self.train_model_table_name = TRAIN_MODEL_TABLE_NAME
        self.model_name = 'lightgbm'
        self.board_type = None
        self.date_start = None
        self.date_end = None
        self.num_rows = None
        
        self.price_limit_rate = None
        self.categorical_features=None
        self.keep_train_model = False
        self.prob_df = pd.DataFrame()
        
    def load_dataset(self, bunch, test_size=None):
        x_values, y_values = bunch.data, bunch.target
        self.categorical_features = bunch.categorical_features
        x_df = pd.DataFrame(x_values,
                            columns = bunch.numeric_features + bunch.categorical_features)
        test_size = test_size if test_size else self.test_size
        x_df[self.categorical_features] = x_df[self.categorical_features].astype('category')
# =============================================================================
#         x_train, x_test, y_train, y_test = train_test_split(x_df,
#                                                             y_values,
#                                                             test_size=test_size)
# =============================================================================
        # 时序数据计算切分位置, 按时间顺序切分数据
        split_idx = int(len(x_df) * (1 - test_size))
        x_train, x_test = x_df[:split_idx], x_df[split_idx:]
        y_train, y_test = y_values[:split_idx], y_values[split_idx:]
        
        x_test = x_test.reset_index(drop=True)  # debug: The index is retained during the train_test_split process. The index is reset in this step.
        return x_train, x_test, y_train, y_test
    
    def dataset_split(self, stock_daily_df):
        """
        Execute feature engineering pipeline and return datasets for training and testing.
        """
        self.board_type = stock_daily_df.board_type.values[0]
        self.price_limit_rate = stock_daily_df.price_limit_rate.values[0]
        logger.info(f'board_type: {self.board_type}| price_limit_rate: {self.price_limit_rate}| df_shape: {stock_daily_df.shape}')
        bunch = self.feature_engineering_pipeline(stock_daily_df)
        x_train, x_test, y_train, y_test = self.load_dataset(bunch)
        return x_train, x_test, y_train, y_test
    
    def train(self, x_train, y_train):
        """
        Train LightGBM model.
        train_model
            primary_key
            model_name:['lightgbm']
            problem_type:['classification','regression']
            task_name:['price_limit','short_term_recommend','stock_pick','stock_recommend']
            board_type:['主板','创业板','科创板','ETF','ST']
            model_path
            date_start
            date_end
            num_rows
            
            insert_timestamp
        """
        del x_train['primary_key']
        model_metadata = {'feature_names': x_train.columns,
                          'categorical_features':self.categorical_features,
                          'target_names':self.target_names,
                          }
        
        lgb_regressor = lgb.LGBMRegressor(**self.params)
        self.model = MultiOutputRegressor(lgb_regressor)
        
        # Fit the model
        self.model.fit(x_train,
                       y_train,
                       #eval_metric='rmse',  # 你可以选择其他评估指标
                       #early_stopping_rounds=50
                       #categorical_feature=self.categorical_features
                       )
        model_dict = vars(self.model.estimator)
        
        #print('model_dict',model_dict)
        #model_dict = {'boosting_type': 'gbdt', 'objective': 'regression', 'num_leaves': 127, 'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 1000, 'subsample_for_bin': 200000, 'min_split_gain': 0.0, 'min_child_weight': 1, 'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 0, 'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'random_state': None, 'n_jobs': None, 'importance_type': 'split', '_Booster': None, '_evals_result': {}, '_best_score': {}, '_best_iteration': -1, '_other_params': {'task': 'train', 'boosting': 'gbdt', 'metric': ['mae'], 'verbose': -1, 'min_child_sample': 40}, '_objective': 'regression', 'class_weight': None, '_class_weight': None, '_class_map': None, '_n_features': -1, '_n_features_in': -1, '_classes': None, '_n_classes': -1, 'task': 'train', 'boosting': 'gbdt', 'metric': ['mae'], 'verbose': -1, 'min_child_sample': 40}
        
        # 指定要提取的字段列表
        selected_fields = ['boosting_type', 'objective', 'num_leaves','max_depth','learning_rate','n_estimators','subsample']
        
        # 使用字典推导式从 model_dict 中提取指定字段的子集
        model_multioutput_selected_dict = {key: value for key, value in model_dict.items() if key in selected_fields}
        
        # 输出提取的字段子集
        #print(model_multioutput_selected_dict)
        
        primary_key = utils_character.generate_random_md5()
        train_model_dict = {
            'primary_key': primary_key,
            'model_name': self.model_name,
            'problem_type': self.problem_type,
            'task_name': self.task_name,
            'board_type': self.board_type,
            'price_limit_rate': self.price_limit_rate,
            'model_path': f'{path}/checkpoint/lightgbm/{primary_key}.joblib',
            'date_start': self.date_start,
            'date_end': self.date_end,
            'num_rows': self.num_rows,
            }
        train_model_dict.update(model_multioutput_selected_dict)
        train_model_pd = pd.json_normalize(train_model_dict)
        train_model_pd['insert_timestamp'] = datetime.now().strftime('%F %T')
        
        with utils_database.engine_conn('postgre') as conn:
            train_model_pd.to_sql(self.train_model_table_name, con=conn.engine, index=False, if_exists='append')
            
        # save_model
        if self.keep_train_model:
            joblib.dump((self.model, model_metadata), train_model_dict['model_path'])
    
    def prediction(self, x_test):
        """
        prediction
        Evaluate model performance, calculate RMSE and MAE, output results to the database, and return DataFrame.
        """
        y_test_pred = self.model.predict(x_test)
        if y_test_pred.ndim > 2: # debug:对于 multiclass 问题，predict() 默认返回每个样本每个类别的概率
            self.prob_df = pd.DataFrame(y_test_pred.reshape(3, y_test_pred.shape[1]).T,columns=['base_index_-1_prob','base_index_0_prob','base_index_1_prob'])
            y_test_pred = np.argmax(y_test_pred, axis=0)
            
        y_test_pred = pd.DataFrame(y_test_pred)
        logger.info(f'y_test_pred: {y_test_pred}')
        return y_test_pred
    
    def field_handle(self, y_test_pred):
        target_names = [x+'_pred' for x in self.target_names]
        y_test_pred.columns = target_names
        return y_test_pred
    
    def train_pipeline(self, subtable):
        self.date_start, self.date_end = subtable['date'].agg(['min', 'max'])
        self.num_rows = subtable.shape[0]
        x_train, x_test, y_train, _ = self.dataset_split(subtable)
        
        self.train(x_train, y_train)
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        y_test_pred = self.prediction(x_test)
        prediction_df = self.field_handle(y_test_pred)
        
        # 通过主键关联字段
        prediction_df['primary_key'] = primary_key_test
        if not self.prob_df.empty:
            prediction_df = pd.concat([prediction_df, self.prob_df], axis=1)
        return prediction_df
    
    def train_board_pipeline(self, stock_daily_df, keep_train_model=False):
        self.keep_train_model = keep_train_model
        prediction_related_df = stock_daily_df.groupby('board_primary_key').apply(self.train_pipeline)
        return prediction_related_df


    
# =============================================================================
#     def plot_feature_importance(X_train):
#         """
#         Plot feature importance using Seaborn.
#     
#         Parameters:
#         - model_multioutput_regressor: The MultiOutputRegressor object.
#         - X_train: The training data used to fit the model.
#         """
#         # Assuming the underlying estimator is a LightGBM model
#         lgb_model = self.model_multioutput_regressor.estimators_[0].estimator_
#     
#         feature_importance_split = pd.DataFrame({
#             'Feature': X_train.columns,  # Assuming X_train is a DataFrame with named columns
#             'Importance': lgb_model.feature_importance(importance_type='split'),
#             'Type': 'split'
#         })
#     
#         feature_importance_gain = pd.DataFrame({
#             'Feature': X_train.columns,
#             'Importance': lgb_model.feature_importance(importance_type='gain'),
#             'Type': 'gain'
#         })
#     
#         feature_importance = pd.concat([feature_importance_split, feature_importance_gain], ignore_index=True)
#     
#         plt.figure(figsize=(12, 6))
#         sns.barplot(x='Importance', y='Feature', data=feature_importance, hue='Type', palette="viridis", dodge=True)
#         plt.title('Feature Importance')
#         plt.show()
# =============================================================================
# =============================================================================
#     def plot_feature_importance(self):
#         """
#         Plot feature importance using Seaborn.
#         """
#         feature_importance_split = pd.DataFrame({
#             'Feature': self.model_multioutput_regressor.feature_name(),
#             'Importance': self.model_multioutput_regressor.feature_importance(importance_type='split'),
#             'Type': 'split'
#         })
#         
#         feature_importance_gain = pd.DataFrame({
#             'Feature': self.model_multioutput_regressor.feature_name(),
#             'Importance': self.model_multioutput_regressor.feature_importance(importance_type='gain'),
#             'Type': 'gain'
#         })
# 
#         feature_importance = pd.concat([feature_importance_split, feature_importance_gain], ignore_index=True)
# 
#         plt.figure(figsize=(12, 6))
#         sns.barplot(x='Importance', y='Feature', data=feature_importance, hue='Type', palette="viridis", dodge=True)
#         plt.title('Feature Importance')
#         plt.show()
#         
# =============================================================================
# =============================================================================
#     def plot_feature_importance(self):
#         """
#         Plot feature importance.
#         """
#         lgb.plot_importance(self.model, importance_type='split', figsize=(10, 6), title='Feature importance (split)')
#         lgb.plot_importance(self.model, importance_type='gain', figsize=(10, 6), title='Feature importance (gain)')
# =============================================================================

# =============================================================================
#     def prediction_y(self, y_test_true, x_test, task_name=None): #, prediction_name=None
#         """
#         evaluate_model
#         Evaluate model performance, calculate RMSE and MAE, output results to the database, and return DataFrame.
#         """
#         ## pred
#         print('x_test', x_test)
#         y_test_pred = self.model_multioutput_regressor.predict(x_test)
#         y_result = pd.DataFrame(np.hstack((y_test_true, y_test_pred)))
#         
#         ## eval
#         #y_test_true = y_test_true.values  # debug
#         print('y_test_true', y_test_true)
#         print('y_test_pred', y_test_pred)
#         mse = mean_squared_error(y_test_true, y_test_pred)
#         rmse = mse ** (0.5)
#         mae = mean_absolute_error(y_test_true, y_test_pred)
#         
#         insert_timestamp = datetime.now().strftime('%F %T')
#         y_eval_dict = {'rmse': round(rmse,3),
#                        'mae': round(mae,3),
#                        'task_name': task_name,
#                        'insert_timestamp': insert_timestamp,
#                        }
#         y_eval_df = pd.DataFrame([y_eval_dict], columns=y_eval_dict.keys())
#         
#         with base_connect_database.engine_conn('postgre') as conn:
#             y_eval_df.to_sql(EVAL_TABLE_NAME, con=conn.engine, index=False, if_exists='append')
#         return y_result
# =============================================================================