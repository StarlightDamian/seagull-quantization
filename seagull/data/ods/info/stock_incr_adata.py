# -*- coding: utf-8 -*-
"""
@Date: 2024/10/28 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_adata.py
@Description: (ods/info/stock_incr_adata)
@Update cycle: day
"""
import os

import adata

from seagull.settings import PATH
from seagull.utils import utils_data, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def get_stock_incr_adata():
    ods_adata_stock_base = adata.stock.info.all_code()
    print(ods_adata_stock_base)  # ['stock_code', 'short_name', 'exchange', 'list_date']
    utils_data.output_database(ods_adata_stock_base,
                               filename='ods_info_stock_incr_adata',
                               if_exists='replace')


if __name__ == '__main__':
    get_stock_incr_adata()
