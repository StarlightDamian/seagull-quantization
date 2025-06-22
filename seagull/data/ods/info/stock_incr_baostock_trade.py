# -*- coding: utf-8 -*-
"""
@Date: 2024/8/13 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock_trade.py
@Description: 获取指定交易日期所有股票列表(ods/info/stock_incr_baostock_trade)
@Update cycle: day
只有交易日才会更新，取当天的不一定及时更新，先尝试前一天
"""
import os
import argparse
from datetime import datetime
# from sqlalchemy import String  # Float, Numeric,

import baostock as bs
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_log, utils_data
from seagull.data import utils_api_baostock

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def _apply_baostock_trade_1(sub):
    date = sub.name
    rs = bs.query_all_stock(day=date)
    trade_sub_df = utils_api_baostock.get_row_data(rs)
    trade_sub_df['date'] = date
    return trade_sub_df


def get_stock_incr_baostock_trade(date_start, date_end=datetime.now().strftime('%F')):
    daily_dates = pd.date_range(start=date_start, end=date_end, freq='D')
    daily_dates_df = pd.DataFrame(daily_dates, columns=['date'])
    
    bs.login()
    trade_df = daily_dates_df.groupby('date').apply(_apply_baostock_trade_1).reset_index(drop=True)
    bs.logout()

    trade_df['date'] = pd.to_datetime(trade_df['date']).dt.strftime('%Y-%m-%d')
    return trade_df


def pipeline(**kwargs):
    trade_df = get_stock_incr_baostock_trade(kwargs)
    logger.info(f'trade_df.shape: {trade_df.shape}')
    utils_data.output_database_large(trade_df, filename='ods_info_stock_incr_baostock_trade')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--date_start', type=str, default='1990-01-01', help='start date')  # 1990-01-01
    parser.add_argument('--date_start', type=str, default='2025-02-28', help='start date')
    parser.add_argument('--date_end', type=str, default=datetime.now().strftime('%F'), help='End date,Default is today')
    args = parser.parse_args()
    
    pipeline(date_start=args.date_start)
    

    