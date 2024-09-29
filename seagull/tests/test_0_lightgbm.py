# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:29:59 2024

@author: awei
lightGBM模型测试(test_0_lightgbm)
"""
from datetime import datetime

import numpy as np
import joblib
import pandas as pd
from sklearn.utils import Bunch

from __init__ import path
from base import base_connect_database
from train import train_0_lightgbm
from feature_engineering import feature_engineering_main


class featureEngineeringTest(feature_engineering_main.featureEngineering):
    def __init__(self, target_pred_names=None):
        self.target_pred_names = target_pred_names
        super().__init__(target_pred_names)
        
    def feature_engineering_pipline(self, history_day_df):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        trading_day_before_dict = self.specified_trading_day_before(pre_date_num=1)
        history_day_df['date_before'] = history_day_df.date.map(trading_day_before_dict)
        #print(history_day_df[['date','date_before']])
        
        trading_day_after_dict = self.specified_trading_day_after(pre_date_num=1)
        history_day_df['date_after'] = history_day_df.date.map(trading_day_after_dict)
        #特征: 微观个股_涨跌停标识
        history_day_df['price_limit'] = history_day_df.apply(lambda row: 1 if row['high'] == row['low'] else 0, axis=1)
        #history_day_df = self.merge_features_after(history_day_df)
        history_day_df = self.build_features_after(history_day_df)
        # history_day_df = self.merge_features_before(history_day_df) # debug
        # 构建数据集
        history_day_df, feature_names = self.build_features_before(history_day_df)
        return history_day_df, feature_names
    
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


class lightgbmTest(train_0_lightgbm.lightgbmTrain):#train_main.stockPickTrain
    def __init__(self, target_pred_names=None):#, date_start, date_end
        """
        Initialize stockPickPrediction object, including feature engineering and database connection.
        """
        super().__init__()
        self.feature_engineering = featureEngineeringTest(target_pred_names)
        self.test_table_name = None
        #self.multioutput_model_path = None
        
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
        print('x_values',x_values)
        print('y_values',y_values)
        x_test = pd.DataFrame(x_values, columns=date_range_bunch.feature_names)
        y_test = pd.DataFrame(y_values, columns=date_range_bunch.target_names)
        return None, x_test, None, y_test
    
    def test_pipline(self, history_day_df, board_type=None, multioutput_model_path=None):
        #print('multioutput_model_path',multioutput_model_path)
        self.multioutput_model_path = multioutput_model_path #if multioutput_model_path!=None
        self.board_type = board_type
        _, x_test, _, y_test = self.feature_engineering_split(history_day_df)
        feature_names, primary_key_name = self.load_model()
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        x_test = x_test.reindex(columns=feature_names, fill_value=False)  # Pop first, then reindex
        y_test_pred = self.prediction(x_test)
        prediction_df = self.field_handle(y_test_pred, x_test)

        # 通过主键关联字段
        related_columns = ['date', 'code', 'code_name', 'preclose', 'isST']
        prediction_df['primary_key'] = primary_key_test
        prediction_related_df = pd.merge(prediction_df, history_day_df[['primary_key']+related_columns], on='primary_key')
        
        with base_connect_database.engine_conn('postgre') as conn:
            prediction_related_df['insert_timestamp'] = datetime.now().strftime('%F %T')
            prediction_related_df.macro_amount_diff_1 = prediction_related_df.macro_amount_diff_1.astype(float)
            prediction_related_df.to_sql(self.test_table_name, con=conn.engine, index=False, if_exists='append')  # 输出到数据库 ,replace
            
        return prediction_related_df
    
    def test_board_pipline(self, history_day_board_df):
        board_model_df = self.board_model()
        board_model_df = board_model_df[board_model_df.task_name==self.task_name][['board_type', 'model_path']]
        prediction_list = []
        #print('history_day_board_df',history_day_board_df)
        for (board_type, model_path) in board_model_df.values:
            print('------------')
            print('board_type', board_type)
            print('model_path', model_path)
            #self.multioutput_model_path = model_path
            history_day_board_df_1 = history_day_board_df[history_day_board_df.board_type==board_type]
            if not history_day_board_df_1.empty:
                #history_day_board_df_1.to_csv(f'{path}/data/history_day_board_df_1.csv',index=False)
                prediction_df_1 = self.test_pipline(history_day_board_df_1, board_type=board_type, multioutput_model_path=model_path)
                prediction_list.append(prediction_df_1)
        prediction_df = pd.concat(prediction_list, axis=0)
        return prediction_df