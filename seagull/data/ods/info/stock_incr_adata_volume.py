# -*- coding: utf-8 -*-
"""
@Date: 2024/10/28 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_adata_volume.py
@Description: 股票股本信息(ods/info/stock_incr_adata_volume)
@Update cycle: day
可以结合当前价格计算市值
"""

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database


def _apply_stock_volume_1(subtable):
    stock_code = subtable.name
    ods_adata_stock_volume_1 = adata.stock.info.get_stock_shares(stock_code=stock_code, is_history=True)
    # ['stock_code', 'change_date', 'total_shares', 'limit_shares','list_a_shares', 'change_reason']
    return ods_adata_stock_volume_1


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
    ods_adata_stock_volume = ods_adata_stock_base.groupby('stock_code').apply(_apply_stock_volume_1)
    print(ods_adata_stock_volume)
    utils_data.output_database(ods_adata_stock_volume,
                               filename='ods_info_incr_adata_stock_volume',
                               if_exists='replace') 
