# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:35:32 2024

@author: awei
lightGBM模型评估(valid_0_lightgbm)
"""
import os
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from seagull.settings import PATH
from train import train_0_lightgbm
from seagull.utils import utils_database, utils_log, utils_data

VALID_TABLE_NAME = 'valid_model'
VALID_DETAILS_TABLE_NAME = 'valid_model_details'
TEST_SIZE = 0
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class LightgbmValid(train_0_lightgbm.LightgbmTrain):
    def __init__(self, target_names=None):
        """
        Initialize stockPickPrediction object, including feature engineering and database connection.
        """
        super().__init__()
        self.test_table_name = None
        #self.model_path = None
        self.valid_tabel_name = VALID_TABLE_NAME
        self.valid_details_tabel_name = VALID_DETAILS_TABLE_NAME
        
        self.primary_key_model = None
        self.test_size = TEST_SIZE
        self.task_name = None
        
    def load_model(self):
        """
        Load model from the specified path.
        """
        self.model_multioutput, model_metadata = joblib.load(self.model_path)
        feature_names = model_metadata['feature_names']
        primary_key_name = model_metadata['primary_key_name']
        return feature_names, primary_key_name
        
    def load_dataset(self, date_range_bunch):
        x_values, y_values = date_range_bunch.data, date_range_bunch.target
        x_test = pd.DataFrame(x_values, columns=date_range_bunch.feature_names)
        y_test = pd.DataFrame(y_values, columns=date_range_bunch.target_names)
        return None, x_test, None, y_test
    
    def _apply_mae_rmse(self, handle_df):
        y_test = handle_df[self.target_real_names].values
        y_pred = handle_df[self.target_names].values
        #pd.DataFrame(y_test).to_csv(f'{PATH}/data/y_test.csv',index=False)
        #pd.DataFrame(y_pred).to_csv(f'{PATH}/data/y_pred.csv',index=False)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return round(mae, 3), round(rmse, 3)
    
    def _apply_valid_1(self, subtable):
        subtable[['mae','rmse']] = self._apply_mae_rmse(subtable)
        return subtable
    
    def valid_pipeline(self, handle_df, board_type=None):
        self.board_type = board_type
        mae, rmse = self._apply_mae_rmse(handle_df)
        # print('mae,rmse',mae,rmse)
        num_rows = handle_df.shape[0]
        date_start, date_end = handle_df['date'].agg(['min', 'max'])
        
        # model_multioutput, model_metadata = joblib.load(self.model_path)
        
        valid_dict = {'primary_key_model': self.primary_key_model,
                      'task_name': self.task_name,
                      'board_type': self.board_type,
                      'date_start': date_start,
                      'date_end': date_end,
                      'num_rows': num_rows,
                      'mae': mae,
                      'rmse': rmse,
                      }
        valid_df = pd.DataFrame.from_records([valid_dict])
        
        valid_details_df = handle_df.groupby('code').apply(self.__apply_valid_1)
        valid_details_df = valid_details_df.drop_duplicates('code', keep='first')
        
        utils_data.output_database(valid_df,
                                   filename = self.valid_tabel_name,
                                   index=False,
                                   if_exists='append'
                                   )
        utils_data.output_database(valid_details_df,
                                   filename = self.valid_details_tabel_name,
                                   index=False,
                                   if_exists='append'
                                   )
        return valid_df, valid_details_df
    
    #def valid_pipline(self, handle_df):
    #    
        #_, x_test, _, y_test = self.feature_engineering_split(daily_df)
    #    valid_df, valid_details_df = self.valid(handle_df)
    #    return valid_df, valid_details_df
    
    def valid_board_pipeline(self, daily_df):
        self.keep_train_model = False
        logger.info(f'daily_df_shape: {daily_df.shape}')
        
        with utils_database.engine_conn("POSTGRES") as conn:
            board_model_df = pd.read_sql(f"""
                                      SELECT
                                          primary_key
                                          ,board_type
                                          ,price_limit_rate
                                          ,model_path
                                      FROM
                                          ads_info_incr_train_model
                                      WHERE 
                                          task_name='{self.task_name}'
                                          """, con=conn.engine)
        board_model_df = board_model_df[(pd.to_datetime(board_model_df.date_end) - pd.to_datetime(board_model_df.date_start)).dt.days>100]
        
        for (primary_key_model, board_type, price_limit_rate, model_path) in board_model_df.values:
            self.primary_key_model = primary_key_model
            self.model_path = model_path
            self.board_type = board_type
            logger.info(f'board_type:{board_type} | price_limit_rate: {price_limit_rate}|model_path:{model_path}')
            board_daily_df = daily_df[(daily_df.board_type==board_type)
                                     &(daily_df.price_limit_rate==price_limit_rate)]
            handle_board_df = self.train_merge(board_daily_df, model_path=self.model_path)
            _, _ = self.valid_pipline(handle_board_df, board_type=board_type)
            
    def train_merge(self, board_daily_df, model_path=None):
        #real_df = self.train_board_pipline(board_daily_df, keep_train_model=False)
        #real_df = real_df[['primary_key'] + self.target_real_names]
        #handle_real_board_df = pd.merge(board_daily_df, real_df, on='primary_key')
        
        board_type = board_daily_df.board_type.values[0]
        pred_df = self.test_stock_pick.test_pipline(board_daily_df, board_type=board_type, model_path=model_path)
        #pred_df = pred_df[['primary_key'] + self.target_names]
        handle_board_df = pd.merge(handle_real_board_df, pred_df, on='primary_key')
        
        #handle_board_df['primary_key'] = self.primary_key_model  # 把主键又日期+代码修改为模型名称
        handle_board_df['primary_key_model'] = self.primary_key_model
        
        handle_board_df = handle_board_df[self.target_real_names + self.target_names + ['primary_key_model', 'code', 'date']]
        return handle_board_df
        
# =============================================================================
#         feature_names, primary_key_name = self.load_model()
#         
#         primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
#         x_test = x_test.reindex(columns=feature_names, fill_value=False)  # Pop first, then reindex
#         y_test_pred = self.prediction(x_test)
#         prediction_df = self.field_handle(y_test_pred, x_test)
#             
# =============================================================================
        