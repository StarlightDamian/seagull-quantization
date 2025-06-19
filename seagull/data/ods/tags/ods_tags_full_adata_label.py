# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:27:03 2024

@author: awei
adata的股票所属板块、概念(ods_flag_full_adata_label)
不能翻墙
"""
import os

import adata
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_data, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class OdsLabel:
    def __init__(self):
        ...
        
    def stock_label_code(self):
        stock_label_code_ths_df = adata.stock.info.all_concept_code_ths()  # 同花顺
        # ['index_code', 'name', 'concept_code', 'source'] 
        
        stock_label_code_east_df = adata.stock.info.all_concept_code_east()  # 东方财富
        # ['concept_code', 'index_code', 'name', 'source']
        
        stock_label_code_df = pd.concat([stock_label_code_ths_df, stock_label_code_east_df], axis=0)
        utils_data.output_database(stock_label_code_df,
                                   filename='ods_flag_full_adata_stock_label_code',
                                   if_exists='replace')
        
    def _apply_stock_label_1(self, subtable):
        index_code = subtable.name  # concept_code有部分为空，尽量用index_code
        concept_code = subtable.concept_code.values[0]
        source = subtable.source.values[0]
        name = subtable['name'].values[0]
        try:
            if source=='同花顺':
                stock_label_df = adata.stock.info.concept_constituent_ths(index_code=index_code, wait_time=15000)
            elif source=='东方财富':
                stock_label_df = adata.stock.info.concept_constituent_east(concept_code, wait_time=10000)
            stock_label_df[['name', 'concept_code', 'index_code', 'source']] = name, concept_code, index_code, source
            utils_data.output_database(stock_label_df,
                                       filename='ods_flag_full_adata_stock_label',
                                       if_exists='append')
            #return df
        except ValueError:
            logger.error(f'concept_code: {concept_code}|index_code: {index_code}| source: {source}')
        
    def stock_label(self):
        with utils_database.engine_conn("POSTGRES") as conn:
            stock_label_code_df = pd.read_sql("ods_flag_full_adata_stock_label_code", con=conn.engine)
        
        stock_label_code_df.groupby('index_code').apply(self._apply_stock_label_1)
        # stock_label_df.columns = ['stock_code', 'short_name', 'name', 'source']
        #utils_data.output_database(stock_label_df,
        #                           filename='ods_flag_full_adata_stock_label',
         #                          if_exists='replace')
        
    def single_stock_label(self):
        # 通过asset_code获取单一股票所属板块、概念
        adata.stock.info.get_plate_east(stock_code="600020", plate_type=1)  # 东方财富
        # ['stock_code', 'plate_code', 'plate_name', 'plate_type', 'source']
        
        df = adata.stock.info.get_concept_baidu(stock_code="600020")  # 百度股市通    
        # ['stock_code', 'concept_code', 'name', 'source', 'reason']
        utils_data.output_database(df,
                                   filename='ods_flag_full_adata_single_stock_label',
                                   if_exists='replace')
        
    def all_index_label(self):
        # 指数
        index_label_df = adata.stock.info.all_index_code()
        # index_label_df.columns = ['index_code', 'concept_code', 'name', 'source']
        utils_data.output_database(index_label_df,
                                   filename='ods_flag_full_adata_all_index_label',
                                   if_exists='replace')
        
    def pipeline(self):
        self.stock_label_code()
        self.stock_label()
        # self.single_stock_label(asset='600020')
        self.all_index_label()
        
        
if __name__ == '__main__':
    ods_label = OdsLabel()
    ods_label.pipeline()