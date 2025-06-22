# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_full_adata_shares.py
@Description: 股票股本信息(ods/info/stock_full_adata_shares)
@Update cycle: day
可以结合当前价格计算市值
"""
import adata
import pandas as pd

from seagull.utils import utils_database, utils_data


def _apply_stock_shares_1(sub):
    """
    获取股票的流通股本和总股本
    """
    stock_code = sub.name
    stock_shares_sub_df = adata.stock.info.get_stock_shares(stock_code=stock_code, is_history=True)
    # ['stock_code', 'change_date', 'total_shares', 'limit_shares','list_a_shares', 'change_reason']
    return stock_shares_sub_df


def pipeline():
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_df = pd.read_sql("ods_info_stock_incr_adata", con=conn.engine)

    stock_shares_df = stock_df.groupby('stock_code').apply(_apply_stock_shares_1)
    utils_data.output_database(stock_shares_df,
                               filename='ods_info_stock_full_adata_shares',
                               if_exists='replace')


if __name__ == '__main__':
    pipeline()
