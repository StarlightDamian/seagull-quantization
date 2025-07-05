# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 23:09:47 2024

@author: awei
(dwd_flag_full_label)
ods_flag_full_baostock_stock_industry
"""
import pandas as pd

from seagull.utils import utils_database, utils_data
from seagull.utils.api import utils_api_adata, utils_api_baostock


class DwdLabel:
    def __init__(self):
        ...
        
    #def dwd_flag_full_label_code(self):
    #    ...
    
    def dwd_flag_full_label(self):
        ...
        
    def pipeline(self):
        ...
        
        
if __name__ == '__main__':
    
    with utils_database.engine_conn("POSTGRES") as conn:
        baostock_stock_label_df = pd.read_sql("ods_flag_full_baostock_stock_label", con=conn.engine)#stock_label_df
        adata_stock_label_df = pd.read_sql("ods_flag_full_adata_stock_label", con=conn.engine)#61163
        
    #adata_stock_label_df.columns='stock_code','full_code','short_name','name','concept_code','index_code', 'source'
    adata_stock_label_df = adata_stock_label_df.rename(columns={'stock_code': 'asset_code',
                                                                'short_name': 'code_name',
                                                                'name': 'label'})
    adata_stock_label_df = utils_api_adata.adata_full_code(adata_stock_label_df)
    adata_stock_label_df = adata_stock_label_df[['full_code', 'asset_code', 'code_name', 'label', 'source']]
    
    
    baostock_stock_label_df = utils_api_baostock.split_baostock_code(baostock_stock_label_df)
    # ['updateDate', 'code', 'code_name', 'industry', 'industryClassification','insert_timestamp', 'market_code', 'asset_code', 'full_code']
    baostock_stock_label_df = baostock_stock_label_df[['code_name', 'industry', 'industryClassification', 'asset_code', 'full_code']]
    baostock_stock_label_df = baostock_stock_label_df.rename(columns={
        'industry': 'label',
        'industryClassification': 'source',
        })
    baostock_stock_label_df = baostock_stock_label_df[['full_code', 'asset_code', 'code_name', 'label', 'source']]
    
    stock_label_df = pd.concat([adata_stock_label_df, baostock_stock_label_df], axis=0)
    stock_label_df = stock_label_df[~((stock_label_df.label=='')|(stock_label_df.label.isnull()))]
    utils_data.output_database(stock_label_df,
                               filename='dwd_tags_full_label',
                               if_exists='replace')
    