# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:47:02 2024

@author: awei
宏观经济-两融(ods_macr_incr_adata_securities_margin_api)
"""
import adata

from __init__ import path
from utils import utils_data

securities_margin_daily_df = adata.sentiment.securities_margin(start_date='1990-01-01')
utils_data.output_database(securities_margin_daily_df,
                           filename='ods_acct_incr_adata_securities_margin',
                           if_exists='replace')
#[3541 rows x 6 columns]
#2010-03-31, 2024-10-29 



# =============================================================================
# trade_date	date	交易日期	2023-07-21
# rzye	decimal	融资余额（元）	1485586705452
# rqye	decimal	融券余额（元）	90400227216
# rzrqye	decimal	融资融券余额（元）	1575986932668
# rzrqyecz	decimal	融资融券余额差值（元）	1575986932668
# =============================================================================
