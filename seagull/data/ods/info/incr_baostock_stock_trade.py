# -*- coding: utf-8 -*-
"""
@Date: 2024/8/13 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: base.py
@Description: 获取指定交易日期所有股票列表(data_ods_info_incr_baostock_trade_stock_api)
只有交易日才会更新，取当天的不一定及时更新，先尝试前一天
"""
# from sqlalchemy import String  # Float, Numeric, 
import os
import argparse
from datetime import datetime

import baostock as bs
import pandas as pd

from __init__ import path
from utils import utils_log, utils_data
from data import  utils_api_baostock

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


def __apply_baostock_trade_stock_1(subtable):
    date = subtable.name
    rs = bs.query_all_stock(day=date)
    baostock_trade_stock_1_df = utils_api_baostock.get_row_data(rs)
    baostock_trade_stock_1_df['date'] = date
    return baostock_trade_stock_1_df

def baostock_trade_stock(date_start, date_end=datetime.now().strftime('%F')):
    daily_dates = pd.date_range(start=date_start, end=date_end, freq='D')
    daily_dates_df = pd.DataFrame(daily_dates,columns=['date'])
    
    bs.login()
    baostock_stock_trade_df = daily_dates_df.groupby('date').apply(__apply_baostock_trade_stock_1).reset_index(drop=True)
    bs.logout()
    
    baostock_stock_trade_df['insert_timestamp'] = datetime.now().strftime("%F %T")
    baostock_stock_trade_df['date'] = pd.to_datetime(baostock_stock_trade_df['date']).dt.strftime('%Y-%m-%d')
    logger.debug(f'baostock_stock_trade_df.shape: {baostock_stock_trade_df.shape}')
    utils_data.output_database_large(baostock_stock_trade_df, filename='ods_info_incr_baostock_stock_trade')
    return baostock_stock_trade_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='1990-01-01', help='start date')  # 1990-01-01
    parser.add_argument('--date_start', type=str, default='2025-02-28', help='start date')
    parser.add_argument('--date_end', type=str, default=datetime.now().strftime('%F'), help='end date')  # today
    args = parser.parse_args()
    
    baostock_trade_stock_df = baostock_trade_stock(date_start=args.date_start)
    

    