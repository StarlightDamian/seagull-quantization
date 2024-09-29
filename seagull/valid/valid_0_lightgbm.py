# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:35:32 2024

@author: awei
lightGBM模型评估(valid_0_lightgbm)
"""
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
from sklearn.utils import Bunch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from __init__ import path
from base import base_connect_database
from train import train_0_lightgbm
from feature_engineering import feature_engineering_main
VALID_TABLE_NAME = 'valid_model'
VALID_DETAILS_TABLE_NAME = 'valid_model_details'
TEST_SIZE = 0

class featureEngineeringValid(feature_engineering_main.featureEngineering):
    def __init__(self, target_pred_names=None):
        self.target_pred_names = target_pred_names
        super().__init__(target_pred_names)
    
    def build_dataset(self, history_day_df, feature_names):
        # debug,通过排除涨跌10%的数据来进行更准确的预测
        #history_day_df = history_day_df[(history_day_df.rear_low_pct_real<=10.5)&(history_day_df.rear_low_pct_real>=-10.5)&(history_day_df.rear_high_pct_real<=10.5)&(history_day_df.rear_high_pct_real>=-10.5)]
        
        # 构建数据集
        feature_names = sorted(feature_names) # 输出有序标签
        # print(f'feature_names_engineering:\n {feature_names}')
        history_day_df[self.target_pred_names] = np.nan
        date_range_dict = {'data': np.array(history_day_df[feature_names].to_records(index=False)),  # 不使用 feature_df.values,使用结构化数组保存每一列的类型
                         'feature_names': feature_names,
                         'target': history_day_df[self.target_pred_names].values,  # 机器学习预测值
                         'target_names': [self.target_pred_names],
                         }
        date_range_bunch = Bunch(**date_range_dict)
        return date_range_bunch


class lightgbmValid(train_0_lightgbm.lightgbmTrain):
    def __init__(self, target_pred_names=None):
        """
        Initialize stockPickPrediction object, including feature engineering and database connection.
        """
        super().__init__()
        self.feature_engineering = featureEngineeringValid(target_pred_names)
        self.test_table_name = None
        #self.multioutput_model_path = None
        self.valid_tabel_name = VALID_TABLE_NAME
        self.valid_details_tabel_name = VALID_DETAILS_TABLE_NAME
        
        self.primary_key_model = None
        self.test_size = TEST_SIZE
        self.task_name = None
        
    def load_model(self):
        """
        Load model from the specified path.
        """
        self.model_multioutput, model_metadata = joblib.load(self.multioutput_model_path)
        feature_names = model_metadata['feature_names']
        primary_key_name = model_metadata['primary_key_name']
        return feature_names, primary_key_name
        
    def load_dataset(self, date_range_bunch):
        x_values, y_values = date_range_bunch.data, date_range_bunch.target
        x_test = pd.DataFrame(x_values, columns=date_range_bunch.feature_names)
        y_test = pd.DataFrame(y_values, columns=date_range_bunch.target_names)
        return None, x_test, None, y_test
    
    def __apply_mae_rmse(self, handle_df):
        y_test = handle_df[self.target_real_names].values
        y_pred = handle_df[self.target_pred_names].values
        #pd.DataFrame(y_test).to_csv(f'{path}/data/y_test.csv',index=False)
        #pd.DataFrame(y_pred).to_csv(f'{path}/data/y_pred.csv',index=False)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return round(mae, 3), round(rmse, 3)
    
    def __apply_valid_1(self, subtable):
        subtable[['mae','rmse']] = self.__apply_mae_rmse(subtable)
        return subtable
    
    def valid_pipline(self, handle_df, board_type=None):
        self.board_type = board_type
        mae, rmse = self.__apply_mae_rmse(handle_df)
        # print('mae,rmse',mae,rmse)
        num_rows = handle_df.shape[0]
        date_start, date_end = handle_df['date'].agg(['min', 'max'])
        
        model_multioutput, model_metadata = joblib.load(self.multioutput_model_path)
        
        valid_dict = {
            'primary_key_model': self.primary_key_model,
            'task_name': self.task_name,
            'board_type': self.board_type,
            'date_start': date_start,
            'date_end': date_end,
            'num_rows': num_rows,
            'mae': mae,
            'rmse': rmse,
            }
        valid_df = pd.DataFrame.from_records([valid_dict])
        valid_df['insert_timestamp'] = datetime.now().strftime('%F %T')
        
        valid_details_df = handle_df.groupby('code').apply(self.__apply_valid_1)
        valid_details_df = valid_details_df.drop_duplicates('code', keep='first')
        valid_details_df['insert_timestamp'] = datetime.now().strftime('%F %T')
        
        with base_connect_database.engine_conn('postgre') as conn:
            valid_df.to_sql(self.valid_tabel_name, con=conn.engine, index=False, if_exists='append')
            valid_details_df.to_sql(self.valid_details_tabel_name, con=conn.engine, index=False, if_exists='append')
            
        return valid_df, valid_details_df
    
    #def valid_pipline(self, handle_df):
    #    
        #_, x_test, _, y_test = self.feature_engineering_split(history_day_df)
    #    valid_df, valid_details_df = self.valid(handle_df)
    #    return valid_df, valid_details_df
    
    
    def valid_board_pipline(self, history_day_df):
        self.keep_train_model = False
        board_model_df = self.board_model()[['primary_key','task_name', 'board_type', 'model_path']]
        print(f'task_name:{self.task_name}')
        board_model_df = board_model_df[board_model_df.task_name==self.task_name]
        
        print('history_day_df_shape',history_day_df.shape)
        for (primary_key_model, _, board_type, model_path) in board_model_df.values:
            self.primary_key_model = primary_key_model
            print(f'board_type:{board_type} |model_path:{model_path}')
            self.multioutput_model_path = model_path
            print('multioutput_model_path:',self.multioutput_model_path)
            self.board_type = board_type
            history_day_board_df = history_day_df[history_day_df.board_type==board_type]
            handle_board_df = self.train_splice(history_day_board_df, multioutput_model_path=self.multioutput_model_path)
            _, _ = self.valid_pipline(handle_board_df, board_type=board_type)
            
            #history_day_board_df.groupby('board_type').apply(self.__apply_eval_board)
            
    def train_splice(self, history_day_board_df, multioutput_model_path=None):
        real_df = self.train_stock_pick.train_board_pipline(history_day_board_df, keep_train_model=False)
        real_df = real_df[['primary_key'] + self.target_real_names]
        handle_real_board_df = pd.merge(history_day_board_df, real_df, on='primary_key')
        
        board_type = history_day_board_df.board_type.values[0]
        pred_df = self.test_stock_pick.test_pipline(history_day_board_df, board_type=board_type, multioutput_model_path=multioutput_model_path)
        pred_df = pred_df[['primary_key'] + self.target_pred_names]
        handle_board_df = pd.merge(handle_real_board_df, pred_df, on='primary_key')
        
        #handle_board_df['primary_key'] = self.primary_key_model  # 把主键又日期+代码修改为模型名称
        handle_board_df['primary_key_model'] = self.primary_key_model
        
        handle_board_df = handle_board_df[self.target_real_names + self.target_pred_names + ['primary_key_model', 'code', 'date']]
        return handle_board_df
# =============================================================================
#         feature_names, primary_key_name = self.load_model()
#         
#         primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
#         x_test = x_test.reindex(columns=feature_names, fill_value=False)  # Pop first, then reindex
#         y_test_pred = self.prediction(x_test)
#         prediction_df = self.field_handle(y_test_pred, x_test)
# 
#         # 通过主键关联字段
#         related_columns = ['date', 'code', 'code_name', 'preclose', 'isST']
#         prediction_df['primary_key'] = primary_key_test
#         prediction_related_df = pd.merge(prediction_df, history_day_df[['primary_key']+related_columns], on='primary_key')
#         
#         self._eval(x_test, y_test, primary_key_test, prediction_related_df)
#         
#         with base_connect_database.engine_conn('postgre') as conn:
#             prediction_related_df['insert_timestamp'] = datetime.now().strftime('%F %T')
#             prediction_related_df.to_sql(self.test_table_name, con=conn.engine, index=False, if_exists='append')  # 输出到数据库 ,replace
#             
# =============================================================================
        