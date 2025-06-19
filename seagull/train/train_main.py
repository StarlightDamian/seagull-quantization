# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:02:54 2024

@author: awei
"""
import argparse

import pandas as pd

from seagull.settings import PATH
from train import train_1_lightgbm_regression, train_1_lightgbm_classification
from base import base_connect_database

PREDICTION_PRICE_OUTPUT_CSV_PATH = f'{PATH}/data/stock_pick_lightgbm.csv'

class stockPickTrain:
    def __init__(self):
        self.standard_stock_picking = train_1_lightgbm_regression.lightgbmRegressionTrain()
        self.stock_price_limit = train_1_lightgbm_classification.lightgbmClassificationTrain()
        
    def stock_pick_pipline(self, history_day_df):
        prediction_stock_price_limit = self.stock_price_limit.data_processing_pipline(history_day_df)
        
        price_limit_df = prediction_stock_price_limit[prediction_stock_price_limit.price_limit_pred==1]
        history_day_df = history_day_df[~(history_day_df.primary_key.isin(price_limit_df.primary_key))]
        history_day_df = history_day_df[~(history_day_df.amount==0)]
        
        prediction_stock_price_related = self.standard_stock_picking.data_processing_pipline(history_day_df)
        return prediction_stock_price_related
    
    
    #{'adjustflag','pbMRQ','pcfNcfTTM','peTTM','price_limit','psTTM','target_date','target_date_pre','tradestatus'}  
    #{'price_limit_pred', 'price_limit_real', 'remarks'}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2018-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2020-01-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # Load date range data
    #history_day_df = data_loading.feather_file_merge(args.date_start, args.date_end)
    with base_connect_database.engine_conn("POSTGRES") as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    stock_pick_train = stockPickTrain()
    prediction_stock_price_related = stock_pick_train.stock_pick_pipline(history_day_df)
    
    # Rename and save to CSV file
    prediction_stock_price_related = prediction_stock_price_related[['rear_low_pct_real', 'rear_low_pct_pred', 'rear_high_pct_real','rear_high_pct_pred','rear_diff_pct_real','rear_diff_pct_pred','rear_open_pct_real','rear_open_pct_pred','rear_close_pct_real','rear_close_pct_pred','rear_rise_pct_real','rear_rise_pct_pred','code','code_name','date','primary_key','open','low','high','close','preclose','isST','remarks','insert_timestamp']]
    prediction_stock_price_related_rename = prediction_stock_price_related.rename(
        columns={'volume': '成交数量',
                 'amount': '成交金额',
                 'turn': '换手率',
                 'pctChg': '涨跌幅',
                 'rear_low_pct_real': '明天_最低价幅_真实值',
                 'rear_low_pct_pred': '明天_最低价幅_预测值',
                 'rear_high_pct_real': '明天_最高价幅_真实值',
                 'rear_high_pct_pred': '明天_最高价幅_预测值',
                 'rear_diff_pct_real': '明天_变化价幅_真实值',
                 'rear_diff_pct_pred': '明天_变化价幅_预测值',
                 'rear_open_pct_real': '明天_开盘价幅_真实值',
                 'rear_open_pct_pred': '明天_开盘价幅_预测值',
                 'rear_close_pct_real': '明天_收盘价幅_真实值',
                 'rear_close_pct_pred': '明天_收盘价幅_预测值',
                 'rear_rise_pct_real': '明天_上升价幅_真实值',
                 'rear_rise_pct_pred': '明天_上升价幅_预测值',
                 'code': '股票代码',
                 'code_name': '股票中文名称',
                 'date': '日期',
                 'primary_key': '信息主键编号',
                 'open': '今开盘价格',
                 'low': '最低价',
                 'high': '最高价',
                 'close': '今收盘价',
                 'preclose': '昨天_收盘价',
                 'isST': '是否ST',
                 'remarks': '备注',
                 'insert_timestamp': '入库时间',
                })
    prediction_stock_price_related_rename.to_csv(PREDICTION_PRICE_OUTPUT_CSV_PATH, index=False)