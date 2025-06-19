# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
测试股价(test_2_reg_price)
"""
import os
import argparse

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log
from tests_ import test_0_lightgbm

TASK_NAME = 'reg_price'
TEST_TABLE_NAME = 'test_2_reg_price'
PATH_CSV = f'{PATH}/_file/test_2_reg_price.csv'

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class TestPrice(test_0_lightgbm.lightgbmTest):
    def __init__(self, multioutput_model_path=None):
        super().__init__()
        self.test_table_name = TEST_TABLE_NAME
        #self.multioutput_model_path = multioutput_model_path
        self.task_name = TASK_NAME
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-12-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for backtesting')
    args = parser.parse_args()

    logger.info(f"""
    task: test_2_reg_price_high
    date_start: {args.date_start}
    date_end: {args.date_end}
    """)
    
    # dataset
    with utils_database.engine_conn("POSTGRES") as conn:
        asset_df = pd.read_sql(f"SELECT * FROM das_wide_incr_train WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    asset_df.drop_duplicates('primary_key', keep='first', inplace=True)
    
    test_price = TestPrice()
    valid_raw_df = test_price.test_board_pipline(asset_df)
    
    # output
    valid_df = pd.merge(valid_raw_df, asset_df[['primary_key','next_high_rate','next_low_rate','next_close_rate','price_limit_rate','open','high','low',
                                                'close','volume','turnover','turnover_pct','chg_rel','date','full_code','code_name']], how='left', on='primary_key')
    valid_df['next_high'] = valid_df['next_high_rate'] * valid_df['close']
    valid_df['next_high_pred'] = valid_df['next_high_rate_pred'] * valid_df['close']
    valid_df['next_low'] = valid_df['next_low_rate'] * valid_df['close']
    valid_df['next_low_pred'] = valid_df['next_low_rate_pred'] * valid_df['close']    
    valid_df['next_close'] = valid_df['next_close_rate'] * valid_df['close']
    valid_df['next_close_pred'] = valid_df['next_close_rate_pred'] * valid_df['close']
    
    valid_df[['next_high','next_low','next_close','next_high_pred','next_low_pred','next_close_pred']] = valid_df[['next_high','next_low','next_close','next_high_pred','next_low_pred','next_close_pred']].round(2)
    valid_df[['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred','next_close_rate','next_close_rate_pred']] = valid_df[['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred','next_close_rate','next_close_rate_pred']].round(4)
    columns_dict = {'open': '开盘价',
                     'high': '最高价',
                     'low': '最低价',
                     'close': '收盘价',
                     'volume': '成交数量',
                     'turnover': '成交金额',
                     'turnover_pct': '换手率',
                     'price_limit_rate': '涨跌停比例',
                     #'macro_value_traded': '深沪成交额',
                     #'macro_value_traded_diff_1': '深沪成交额增量',
                     'chg_rel': '涨跌幅',
                     'next_low': '明天_最低价_真实值',
                     'next_low_pred': '明天_最低价_预测值',
                     'next_low_rate': '明天_最低价幅_真实值',
                     'next_low_rate_pred': '明天_最低价幅_预测值',
                     'next_high': '明天_最高价_真实值',
                     'next_high_pred': '明天_最高价_预测值',
                     'next_high_rate': '明天_最高价幅_真实值',
                     'next_high_rate_pred': '明天_最高价幅_预测值',
                     'next_close': '明天_收盘价_真实值',
                     'next_close_pred': '明天_收盘价_预测值',
                     'next_close_rate': '明天_收盘价幅_真实值',
                     'next_close_rate_pred': '明天_收盘价幅_预测值',
                     'date': '日期',
                     'full_code': '股票代码',
                     'code_name': '公司名称',
                    }
    output_valid_df = valid_df.rename(columns=columns_dict)
    output_valid_df = output_valid_df[columns_dict.values()]
    output_valid_df.to_csv(PATH_CSV, index=False)