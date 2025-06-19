# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock/full/adata_shares.py
@Description:
"""
import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database,utils_log, utils_data


def dwd_stock_shares(sub):
    """
    获取股票的流通股本和总股本
    """
    stock_code = sub.name
    stock_shares_sub_df = adata.stock.info.get_stock_shares(stock_code=stock_code, is_history=True)
    return stock_shares_sub_df


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_df = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)

    stock_shares_df = stock_df.groupby('stock_code').apply(dwd_stock_shares)
    utils_data.output_database(stock_shares_df,
                               filename='ods_info_full_adata_stock_shares',
                               if_exists='replace')
