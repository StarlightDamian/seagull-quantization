# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:22:00 2025

@author: Damian
baostock行业分类(ods_flag_full_baostock_stock_label)
"""
import baostock as bs

from seagull.settings import PATH
from seagull.utils import utils_data
from seagull.data import utils_api_baostock


def query_stock_industry():
    bs.login()
    rs = bs.query_stock_industry()
    bs.logout()
    result = utils_api_baostock.get_row_data(rs)
    utils_data.output_database(result,
                               filename='ods_flag_full_baostock_stock_label',
                               )


if __name__ == '__main__':
    query_stock_industry()
    