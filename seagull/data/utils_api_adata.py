# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:52:59 2024

@author: awei
adata的api数据处理工具(utils_api_adata)
"""

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database


def adata_full_code(stock_df):
    with utils_database.engine_conn("POSTGRES") as conn:
        df = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
    df = df.rename(columns={'stock_code': 'asset_code',
                            'exchange': 'market_code'})
    df.market_code = df.market_code.str.lower()
    df['full_code'] = df['asset_code'] + '.' + df['market_code']
    stock_df = pd.merge(stock_df, df[['asset_code', 'full_code']], on='asset_code', how='left')
    return stock_df
    
if __name__ == '__main__': 
    with utils_database.engine_conn("POSTGRES") as conn:
        adata_stock_label_df = pd.read_sql("ods_flag_full_adata_stock_label", con=conn.engine)
    result_df = adata_full_code(adata_stock_label_df)
    print(result_df)