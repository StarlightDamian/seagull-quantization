# -*- coding: utf-8 -*-
"""
@Date: 2025/5/7 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_baostock.py
@Description: baostock证券资料(ods/info/stock_incr_baostock)
@Update cycle: day
"""
import baostock as bs

from seagull.utils import utils_data
from seagull.utils.api import utils_api_baostock


def get_stock_incr_baostock():
    bs.login()
    rs = bs.query_stock_basic()
    bs.logout()
    result = utils_api_baostock.get_row_data(rs)
    utils_data.output_database(result,
                               filename='ods_info_stock_incr_baostock',
                               )


if __name__ == '__main__':
    get_stock_incr_baostock()
