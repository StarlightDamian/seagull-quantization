# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:47:48 2024

@author: awei

dwd_part_full_index_base
"""
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database
from seagull.data import utils_api_baostock

if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        index_df = pd.read_sql("ods_part_full_baostock_index", con=conn.engine)
        
    index_df = utils_api_baostock.split_baostock_code(index_df)
    index_df = index_df.rename(columns={'updateDate': 'update_date'})
    
    index_df = index_df[[ 'full_code', 'asset_code', 'market_code','index', 'index_name', 'update_date']]
    utils_data.output_database(index_df,
                               filename='dwd_part_full_index_base',
                               if_exists='replace')
