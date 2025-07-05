# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:29:31 2024

@author: awei
龙虎榜(ods_acct_efinance_hot_api)
"""
import efinance as ef
import pandas as pd
from seagull.settings import PATH
from seagull.utils import utils_data, utils_database


if __name__ == '__main__':
    start_date = '1990-01-20' # 开始日期
    end_date = '2024-10-31' # 结束日期
    with utils_database.engine_conn("POSTGRES") as conn:
        ods_adata_stock_base = pd.read_sql("ods_info_incr_adata_stock_base", con=conn.engine)
        
        ods_capital_flow = pd.read_sql("ods_feat_incr_adata_capital_flow", con=conn.engine)

# =============================================================================
#     hot_df = ef.stock.get_daily_billboard(start_date = start_date,end_date = end_date)
#     print(hot_df)
#     utils_data.output_database(hot_df,
#                                filename='ods_acct_efinance_hot',
#                                if_exists='replace',
#                                index=False)
# =============================================================================
