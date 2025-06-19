# -*- coding: utf-8 -*-
"""
@Date: 2025/5/7 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: base.py
@Description: baostock证券资料(ods_info_incr_baostock_stock_base)
"""
import baostock as bs

from __init__ import path
from utils import utils_data 
from data import utils_api_baostock


def query_stock_basic():
    bs.login()
    rs = bs.query_stock_basic()
    bs.logout()
    result = utils_api_baostock.get_row_data(rs)
    utils_data.output_database(result,
                               filename='ods_info_incr_baostock_stock_base',
                               )
    
if __name__ == '__main__':
    query_stock_basic()
