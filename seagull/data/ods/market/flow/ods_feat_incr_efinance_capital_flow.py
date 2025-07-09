# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:45:33 2025

@author: awei
(ods_feat_incr_efinance_capital_flow_api)
"""
import os

import pandas as pd
import efinance as ef

from seagull.settings import PATH
from seagull.utils import utils_database, utils_data, utils_thread, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


def _apply_stock_capital_flow_1(sub):
    #df = ef.stock.get_history_bill(stock_code='000004')
    stock_code = sub['stock_code'].iloc[0] # sub.name
    try:
        df = ef.stock.get_history_bill(stock_code=stock_code)
        return df
    except Exception as e:
        logger.error(f'{stock_code} {e}')

if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
        #ods_capital_flow = pd.read_sql("ods_feat_incr_adata_capital_flow", con=conn.engine)
        #ods_adata_stock_base = ods_adata_stock_base[~(ods_adata_stock_base.stock_code.isin(ods_capital_flow.stock_code))]
    
    
    grouped = ods_adata_stock_base.groupby('stock_code')
    df = utils_thread.thread(grouped, _apply_stock_capital_flow_1, max_workers=8)
    
    
    utils_data.output_database_large(df,
                               filename='ods_feat_incr_efinance_capital_flow',
                               if_exists='append')
    
    