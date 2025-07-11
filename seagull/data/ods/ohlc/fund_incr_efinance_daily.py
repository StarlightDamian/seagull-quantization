# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 18:28
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_incr_efinance_daily.py
@Description: 
"""
import pandas as pd
import efinance as ef
from seagull.utils import utils_time, utils_data, utils_thread, utils_database, utils_character, utils_log

# ETF 代码（以中概互联网 ETF 为例）
with utils_database.engine_conn("POSTGRES") as conn:
    asset_code_df = pd.read_sql(
        "SELECT 基金代码 as asset_code FROM ods_info_fund_full_efinance where 基金类型 = 'etf'",
        con=conn.engine)
    fund_df = pd.read_sql("select distinct 股票代码 as asset_code from ods_ohlc_fund_incr_efinance_daily",
                          con=conn.engine)

asset_code_list = asset_code_df[~(asset_code_df['asset_code'].isin(fund_df.asset_code))]['asset_code'].tolist()
print(len(asset_code_list))
fund_df_dict = ef.stock.get_quote_history(asset_code_list, klt=101, fqt=1, beg="19000101", end="20500101")
fund_df = pd.concat({asset_code: df for asset_code, df in fund_df_dict.items() if not df.empty}, ignore_index=True)
fund_df['freq_code'] = 101  # 天级
fund_df['adj_code'] = 1  # 前复权
utils_data.output_database_large(fund_df,
                                 filename='ods_ohlc_fund_incr_efinance_daily',
                                 if_exists='append'
                                 )
