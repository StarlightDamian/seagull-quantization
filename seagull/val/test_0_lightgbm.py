# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:29:59 2024

@author: awei
lightGBM模型测试(test_0_lightgbm)
"""
import os
#from datetime import datetime

#import numpy as np
import joblib
import pandas as pd
#from sklearn.utils import Bunch

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log#, utils_data
from train import train_0_lightgbm

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class lightgbmTest(train_0_lightgbm.LightgbmTrain):
    def __init__(self):
        """
        Initialize stockPickPrediction object, including feature engineering and database connection.
        """
        super().__init__()
        self.test_table_name = None
        
    def load_model(self):
        """
        Load model from the specified path.
        """
        self.model, model_metadata = joblib.load(self.model_path)
        feature_names = model_metadata['feature_names']
        self.categorical_features = model_metadata['categorical_features']
        self.target_names = model_metadata['target_names']
        return feature_names
        
    def load_dataset(self, bunch):
        x_values = bunch.data
        x_df = pd.DataFrame(x_values,
                            columns = bunch.numeric_features + bunch.categorical_features)
        x_df[self.categorical_features] = x_df[self.categorical_features].astype('category')
        x_test = pd.DataFrame(x_df, 
                              columns=bunch.numeric_features + bunch.categorical_features)
        return None, x_test, None, None
    
    def test_pipline(self, daily_df, board_type=None, model_path=None):
        self.model_path = model_path
        self.board_type = board_type
        feature_names = self.load_model()
        _, x_test, _, _ = self.dataset_split(daily_df)
        
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        x_test = x_test.reindex(columns = feature_names,
                                fill_value = False)  # Pop first, then reindex
        y_test_pred = self.prediction(x_test)
        prediction_df = self.field_handle(y_test_pred)

        # 通过主键关联字段
        prediction_df['primary_key'] = primary_key_test
        if not self.prob_df.empty:
            prediction_df = pd.concat([prediction_df, self.prob_df], axis=1)
# =============================================================================
#         utils_data.output_database(prediction_df,
#                                    filename=self.test_table_name,
#                                    index=False,
#                                    if_exists='append')
# =============================================================================
        return prediction_df
    
    def test_board_pipline(self, daily_df):
        print('task_name',self.task_name)
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
        prediction_list = []
        print(f'board_model_df:{board_model_df}')
        for (primary_key, board_type,price_limit_rate, model_path) in board_model_df.values:
            logger.info(f'board_type:{board_type} | price_limit_rate: {price_limit_rate}|model_path:{model_path}')
            board_daily_df = daily_df[(daily_df.board_type==board_type)
                                     &(daily_df.price_limit_rate==price_limit_rate)]
            if not board_daily_df.empty:
                board_prediction_df = self.test_pipline(board_daily_df, board_type=board_type, model_path=model_path)
                prediction_list.append(board_prediction_df)
        prediction_df = pd.concat(prediction_list, axis=0)
        return prediction_df
    