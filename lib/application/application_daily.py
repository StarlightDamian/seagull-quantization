# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:58:16 2023

@author: awei
应用层_每日定时任务(application_daily)
"""
import argparse
import schedule
from loguru import logger

import baostock as bs

from __init__ import path
from data import data_ods_a_stock_k
from data.data_ods import odsData
ods_data = odsData()

def custom_date(date=None):
    """
    功能：Directly execute daily tasks for the day
    """
    try:
        bs.login()
        
        ## ODS层
        ods_data.api_baostock(data_type='交易日')
        ods_data.api_baostock(data_type='行业分类')
        ods_data.api_baostock(data_type='证券资料')
        ods_data.api_baostock(data_type='证券代码')
        ods_data.api_adata(data_type='ETF代码')
        ods_data.api_efinance(data_type='ETF日频')
        # get_data.api_efinance(data_type='ETF五分钟频')
        
        # 每日k线
        get_day_data = data_ods_a_stock_k.GetDayData()
        get_day_data.add_new_data()
        
        ## DWD层
        
    except Exception as e:
        logger.error(f"Exception when logging in to obtain transaction day data: {e}")
    finally:
        bs.logout()

def time():
    """
    功能：任务定时
    """
    schedule.every().day.at("17:18").do(custom_date)
    while True:
        schedule.run_pending()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='now')  # 默认daily每日离线任务,可选['daily','now','2022-09-13']
    args = parser.parse_args()
    
    if args.date not in ['daily', 'now']:
        custom_date(args.date)
    elif args.date == 'now':
        custom_date()
    elif args.date == 'daily':
        time()




