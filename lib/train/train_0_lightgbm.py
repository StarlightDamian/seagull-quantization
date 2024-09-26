# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:53:35 2024

@author: awei
lightGBM模型训练(train_0_lightgbm)
"""
from datetime import datetime
from loguru import logger

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from __init__ import path
from feature_engineering import feature_engineering_main
from base import base_connect_database, base_utils

TEST_SIZE = 0.15 #数据集分割中测试集占比

TRAIN_MODEL_TABLE_NAME = 'train_model'

class lightgbmTrain:
    def __init__(self, TARGET_REAL_NAMES=None):
        
        self.train_table_name = None
        self.multioutput_model_path = None
        self.price_limit_pct = None
        self.feature_engineering = feature_engineering_main.featureEngineering(TARGET_REAL_NAMES)
        self.model_multioutput = None
        self.test_size = TEST_SIZE
        
        #train_model
        self.train_model_table_name = TRAIN_MODEL_TABLE_NAME
        self.model_name = 'lightgbm'
        self.board_type = None
        self.date_start = None
        self.date_end = None
        self.num_rows = None
        
        self.keep_train_model = False
        
    def load_dataset(self, date_range_bunch, test_size=None):
        x_values, y_values = date_range_bunch.data, date_range_bunch.target
        x_df = pd.DataFrame(x_values, columns=date_range_bunch.feature_names)
        test_size = test_size if test_size else self.test_size
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_values, test_size=test_size)
        x_test = x_test.reset_index(drop=True)  # The index is retained during the train_test_split process. The index is reset in this step.
        return x_train, x_test, y_train, y_test
    
    def feature_engineering_split(self, history_day_df):
        """
        Execute feature engineering pipeline and return datasets for training and testing.
        """
        logger.info('aaa',history_day_df.shape)
        date_range_bunch = self.feature_engineering.feature_engineering_dataset_pipline(history_day_df, price_limit_pct=self.price_limit_pct)
        logger.info('date_range_bunch',date_range_bunch)
        logger.info('history_day_df0',history_day_df.shape)
        x_train, x_test, y_train, y_test = self.load_dataset(date_range_bunch)
        return x_train, x_test, y_train, y_test
    
    def field_handle(self, y_test_pred, x_test):
        y_test_pred.columns = self.target_pred_names
        prediction_df  = pd.concat([y_test_pred, x_test], axis=1)
        
        prediction_df['remarks'] = ''
        prediction_df = prediction_df[self.target_pred_names+ ['open','high', 'low', 'close','volume', 'amount','turn', 'macro_amount', 'macro_amount_diff_1', 'pctChg', 'remarks']]
        
        return prediction_df
    
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
        #print('x_train',x_train)
        # feature
        model_metadata = {'primary_key_name': 'primary_key'}
        del x_train[model_metadata['primary_key_name']]
        model_metadata['feature_names'] = x_train.columns
        
        # fitting the model
        #x_train.to_csv(f'{path}/data/train_short_term_recommend_x_train.csv',index=False)
        #pd.DataFrame(y_train).to_csv(f'{path}/data/train_short_term_recommend_y_train.csv',index=False)
        self.model_multioutput.fit(x_train, y_train)
        
        model_multioutput_dict = vars(self.model_multioutput.estimator)
        
        #print('model_multioutput_dict',model_multioutput_dict)
        #model_multioutput_dict = {'boosting_type': 'gbdt', 'objective': 'regression', 'num_leaves': 127, 'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 1000, 'subsample_for_bin': 200000, 'min_split_gain': 0.0, 'min_child_weight': 1, 'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 0, 'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'random_state': None, 'n_jobs': None, 'importance_type': 'split', '_Booster': None, '_evals_result': {}, '_best_score': {}, '_best_iteration': -1, '_other_params': {'task': 'train', 'boosting': 'gbdt', 'metric': ['mae'], 'verbose': -1, 'min_child_sample': 40}, '_objective': 'regression', 'class_weight': None, '_class_weight': None, '_class_map': None, '_n_features': -1, '_n_features_in': -1, '_classes': None, '_n_classes': -1, 'task': 'train', 'boosting': 'gbdt', 'metric': ['mae'], 'verbose': -1, 'min_child_sample': 40}
        
        # 指定要提取的字段列表
        selected_fields = ['boosting_type', 'objective', 'num_leaves','max_depth','learning_rate','n_estimators','subsample']
        
        # 使用字典推导式从 model_multioutput_dict 中提取指定字段的子集
        model_multioutput_selected_dict = {key: value for key, value in model_multioutput_dict.items() if key in selected_fields}
        
        # 输出提取的字段子集
        #print(model_multioutput_selected_dict)

        primary_key = base_utils.generate_random_md5()
        train_model_dict = {
            'primary_key': primary_key,
            'problem_type': self.problem_type,
            'model_name': self.model_name,
            'task_name': self.task_name,
            'board_type': self.board_type,
            'model_path': f'{path}/checkpoint/lightgbm/{primary_key}.joblib',
            'date_start': self.date_start,
            'date_end': self.date_end,
            'num_rows': self.num_rows,
            }
        train_model_dict.update(model_multioutput_selected_dict)
        train_model_pd = pd.json_normalize(train_model_dict)
        train_model_pd['insert_timestamp'] = datetime.now().strftime('%F %T')
        
        if self.keep_train_model:
            with base_connect_database.engine_conn('postgre') as conn:
                train_model_pd.to_sql(self.train_model_table_name, con=conn.engine, index=False, if_exists='append')
            
        # save_model
        if self.keep_train_model:
            joblib.dump((self.model_multioutput, model_metadata), train_model_dict['model_path'])
    
    def prediction(self, x_test):
        """
        prediction
        Evaluate model performance, calculate RMSE and MAE, output results to the database, and return DataFrame.
        """
        # x_test.to_csv(f'{path}/data/recommended_test.csv',index=False)
        # logger.info('x_test',x_test)
        y_test_pred = self.model_multioutput.predict(x_test)
        y_test_pred = pd.DataFrame(y_test_pred)
        return y_test_pred
    
    def train_pipline(self, history_day_df, board_type=None, price_limit_pct=None):
        self.date_start, self.date_end = history_day_df['date'].agg(['min', 'max'])
        self.num_rows = history_day_df.shape[0]
        self.board_type = board_type
        self.price_limit_pct = price_limit_pct
        x_train, x_test, y_train, y_test = self.feature_engineering_split(history_day_df)
        self.train(x_train, y_train)
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        y_test_pred = self.prediction(x_test)
        prediction_df = self.field_handle(y_test_pred, x_test)
        
        # 通过主键关联字段
        related_columns = ['date', 'code', 'code_name', 'preclose', 'isST']
        prediction_df['primary_key'] = primary_key_test
        prediction_related_df = pd.merge(prediction_df, history_day_df[['primary_key']+related_columns], on='primary_key')
        
        with base_connect_database.engine_conn('postgre') as conn:
            prediction_related_df['insert_timestamp'] = datetime.now().strftime('%F %T')
            prediction_related_df.to_sql(self.train_table_name, con=conn.engine, index=False, if_exists='replace')
        
        return prediction_related_df

    def board_data(self, history_day_df):
        """
        history_day_df.board_type.value_counts()
        主板     651443
        创业板    180303
        指数     121512
        ST      21363
        科创板      4527
        """
        with base_connect_database.engine_conn('postgre') as conn:
            all_stock_copy_df = pd.read_sql('all_stock_copy', con=conn.engine)
        history_day_df.drop_duplicates('primary_key',keep='first', inplace=True)
        all_stock_copy_df.drop_duplicates('code',keep='first', inplace=True)
        
        all_stock_copy_df = all_stock_copy_df[['code', 'board_type', 'price_limit_pct']]
        history_day_df = pd.merge(history_day_df, all_stock_copy_df, on='code')
        
        # ST数据
        history_day_df.loc[history_day_df.isST=='1', ['board_type', 'price_limit_pct']] = 'ST', 5
        return history_day_df
    
    def board_model(self):
        with base_connect_database.engine_conn('postgre') as conn:
            train_model = pd.read_sql('train_model', con=conn.engine)
        
        #train_model.drop(columns=['primary_key'], inplace=True)
        
        board_model_df = train_model.loc[(train_model.task_name==self.task_name)
                                      &((pd.to_datetime(train_model.date_end) - pd.to_datetime(train_model.date_start)).dt.days>100)]
        
        #print('board_model_df',board_model_df)
        return board_model_df
    
    def __apply_train_board_1(cls, subtable):
        board_type = subtable.name  # board_type='主板'
        logger.info(f'--------\n板块:{board_type}')
        price_limit_pct = subtable.price_limit_pct.values[0]
        prediction_1_df = cls.train_pipline(subtable, board_type=board_type, price_limit_pct=price_limit_pct)
        return prediction_1_df
    
    def train_board_pipline(self, history_day_df, keep_train_model=False):
        self.keep_train_model = keep_train_model
        print('========')
        logger.info(f'model:{self.task_name}')
        prediction_df = history_day_df.groupby('board_type').apply(self.__apply_train_board_1)
        return prediction_df
    
        #history_day_not_st_df = history_day_df[~(history_day_df.isST=='1')]
        #history_day_st_df = history_day_df[history_day_df.isST=='1']
# =============================================================================
# import numpy as np
# from sklearn.metrics import make_scorer
# 
# # 定义自定义评估指标
# def custom_eval_metric(y_true, y_pred):
#     # 根据实际情况修改这里的限制范围和惩罚方式
#     lower_bound = -0.1  # 下限
#     upper_bound = 0.1   # 上限
#     penalty = 0.5       # 惩罚系数
# 
#     # 对预测值进行惩罚，超出范围的部分乘以惩罚系数
#     y_pred_penalty = np.where(y_pred < lower_bound, (y_pred - lower_bound) * penalty + lower_bound, y_pred)
#     y_pred_penalty = np.where(y_pred_penalty > upper_bound, (y_pred_penalty - upper_bound) * penalty + upper_bound, y_pred_penalty)
# 
#     # 返回自定义评估指标，例如 RMSE
#     return np.sqrt(np.mean((y_true - y_pred_penalty) ** 2))
# 
# # 创建自定义评估指标的 Scorer
# custom_scorer = make_scorer(custom_eval_metric, greater_is_better=False)
# 
# # 在模型训练时使用自定义评估指标
# model.fit(X_train, y_train, eval_metric=custom_eval_metric)
# =============================================================================

# =============================================================================
# RMSE (Root Mean Squared Error):
#     优点：对较大的误差更加敏感，因为它平方了每个误差值。
#     缺点：对于异常值更为敏感，因为平方可能会放大异常值的影响。
# MAE (Mean Absolute Error):
#     优点：对异常值不敏感，因为它使用的是误差的绝对值。
#     缺点：不像 RMSE 那样对大误差给予更大的权重。
# 选择哪个指标通常取决于你对模型误差的偏好。如果你更关注大误差，可能会选择使用 RMSE。如果你希望对所有误差都保持相对平等的关注，那么 MAE 可能是更好的选择。
# =============================================================================
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