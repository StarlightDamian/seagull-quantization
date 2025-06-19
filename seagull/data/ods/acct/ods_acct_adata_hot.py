# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:56:34 2024

@author: awei
龙虎榜(ods_acct_adata_hot_api)
"""

import adata
from seagull.settings import PATH
from seagull.utils import utils_data

def _apply_hot_1(subtable):
    stock_code = subtable.name
    df = adata.sentiment.hot.list_a_list_daily(report_date='2024-07-04')


if __name__ == '__main__':
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
        ods_capital_flow = pd.read_sql("ods_feat_incr_adata_capital_flow", con=conn.engine)
        ods_adata_stock_base = ods_adata_stock_base[~(ods_adata_stock_base.stock_code.isin(ods_capital_flow.stock_code))]
    ods_adata_stock_base.groupby('stock_code').apply(_apply_stock_capital_flow_1)
    #print(ods_adata_capital_flow)
    
    
    
    adata.sentiment.hot.list_a_list_daily(report_date='2005-06-03')
    
    
    sentiment.hot.pop_rank_100_east()
    sentiment.hot.hot_rank_100_ths()	