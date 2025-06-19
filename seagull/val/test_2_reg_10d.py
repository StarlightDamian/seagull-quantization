# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:50:27 2024

@author: awei
测试10日股价(test_2_reg_10d)

"""
import os
import argparse

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log
from tests_ import test_0_lightgbm

TASK_NAME = 'reg_price'
TEST_TABLE_NAME = 'test_2_reg_price'
PATH_CSV = f'{PATH}/data/test_2_reg_price_20250221.csv'

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
    #parser.add_argument('--date_start', type=str, default='2024-09-15', help='Start time for backtesting')
    parser.add_argument('--date_start', type=str, default='2025-01-10', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2027-01-01', help='End time for backtesting')
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
    
    ## output
    valid_df = pd.merge(valid_raw_df,
                        asset_df[['primary_key','next_high_rate','next_low_rate','y_10d_low_rate','y_10d_high_rate',
                                  'y_10d_vwap_rate','y_10d_max_dd','price_limit_rate','open','high','low',
                                  'close','volume','turnover','turnover_pct','chg_rel','date','full_code','code_name']],
                        how='left',
                        on='primary_key')
    
    # 最高价
    valid_df['next_high'] = valid_df['next_high_rate'] * valid_df['close']
    valid_df['next_high_pred'] = valid_df['next_high_rate_pred'] * valid_df['close']
    
    # 最低价
    valid_df['next_low'] = valid_df['next_low_rate'] * valid_df['close']
    valid_df['next_low_pred'] = valid_df['next_low_rate_pred'] * valid_df['close']
    
    # 10日vwap
    valid_df['y_10d_vwap'] = valid_df['y_10d_vwap_rate'] * valid_df['close']
    valid_df['y_10d_vwap_pred'] = valid_df['y_10d_vwap_rate_pred'] * valid_df['close']
    
    # 10日回撤
    # valid_df['y_10d_max_dd_pred'] = valid_df['y_10d_max_dd'] * valid_df['close']
    # valid_df['y_10d_max_dd_pred'] = valid_df['y_10d_max_dd_pred'] * valid_df['close']
    
    # 10日vwap回撤比
    valid_df['y_10d_vwap_drawdown_rate'] = (valid_df['y_10d_vwap_rate_pred'] / (valid_df['y_10d_max_dd_pred'] + 1))
    valid_df['y_10d_vwap_drawdown_pct'] = valid_df['y_10d_vwap_drawdown_rate'] * 100
    
    # 10日最低价
    valid_df['y_10d_low'] = valid_df['y_10d_low_rate'] * valid_df['close']
    valid_df['y_10d_low_pred'] = valid_df['y_10d_low_rate_pred'] * valid_df['close']
    
    # 10日最高价
    valid_df['y_10d_high'] = valid_df['y_10d_high_rate'] * valid_df['close']
    valid_df['y_10d_high_pred'] = valid_df['y_10d_high_rate_pred'] * valid_df['close']
    
    # 小数位数
    round_2_columns = ['y_10d_vwap_drawdown_pct']
    valid_df[round_2_columns] = valid_df[round_2_columns].round(2)
    
    round_3_columns = ['next_high','next_low','next_high_pred','next_low_pred','y_10d_vwap','y_10d_vwap_pred',
                       'y_10d_low','y_10d_low_pred','y_10d_high','y_10d_high_pred']
    valid_df[round_3_columns] = valid_df[round_3_columns].round(3)
    
    round_4_columns = ['next_high_rate','next_high_rate_pred','next_low_rate','next_low_rate_pred','y_10d_max_dd',
                       'y_10d_max_dd_pred']
    valid_df[round_4_columns] = valid_df[round_4_columns].round(4)
    
    valid_df = valid_df.sort_values(by='y_10d_vwap_drawdown_pct' ,ascending=False)
    
    columns_dict = {'date': '日期',
                    'full_code': '股票代码',
                    'code_name': '公司名称',
                    'open': '开盘价',
                    'high': '最高价',
                    'low': '最低价',
                    'close': '收盘价',
                    'volume': '成交数量',
                    'turnover': '成交金额',
                    'turnover_pct': '换手率',
                    'next_low': '明天_最低价_真实值',
                    'next_low_pred': '明天_最低价_预测值',
                    'next_low_rate': '明天_最低价幅_真实值',
                    'next_low_rate_pred': '明天_最低价幅_预测值',
                    'next_high': '明天_最高价_真实值',
                    'next_high_pred': '明天_最高价_预测值',
                    'next_high_rate': '明天_最高价幅_真实值',
                    'next_high_rate_pred': '明天_最高价幅_预测值',
                    'y_10d_vwap': '10日_平均成本_真实值',
                    'y_10d_vwap_pred': '10日_平均成本_预测值',
                    'y_10d_max_dd': '10日_回撤_真实值',
                    'y_10d_max_dd_pred': '10日_回撤_预测值',
                    'y_10d_low': '10日_最低价_真实值',
                    'y_10d_low_pred': '10日_最低价_预测值',
                    'y_10d_high': '10日_最高价_真实值',
                    'y_10d_high_pred': '10日_最高价_预测值',
                    'y_10d_vwap_drawdown_pct': '单位风险收益_预测值',
                    'price_limit_rate': '涨跌停比例',
                    #'macro_value_traded': '深沪成交额',
                    #'macro_value_traded_diff_1': '深沪成交额增量',
                    #'chg_rel': '涨跌幅',
                    #'next_close_rate_pred',
                    #'y_10d_vwap_rate_pred',
                    #'y_10d_high_rate_pred',
                    #'y_10d_low_rate_pred',
                    #'primary_key',
                    #'y_10d_low_rate',
                    #'y_10d_high_rate',
                    #'y_10d_vwap_rate',
                    }
    output_valid_df = valid_df.rename(columns=columns_dict)
    output_valid_df = output_valid_df[columns_dict.values()]
    output_valid_df.to_csv(PATH_CSV, index=False)
    
    # valid_df[['y_10d_vwap_rate_pred','y_10d_max_dd_pred','y_10d_vwap_drawdown_pct']]
    # valid_df.loc[valid_df['y_10d_vwap_drawdown_pct'].idxmax()]
    