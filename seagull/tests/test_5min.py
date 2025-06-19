# 每2000次访问暂停10分钟
# import efinance as ef
# df = ef.stock.get_quote_history(['600418', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009'],
#                                 #date_start='2025-01-01',
#                                 #date_end='2026-01-01',
#                                 klt=5)
# print(df)
import efinance as ef
stock_code = '300028'
# 5 分钟
frequency = 5
df = ef.stock.get_quote_history(stock_code, klt=frequency)
print(df)
# import os
# import argparse
# from datetime import datetime
#
# import pandas as pd
# import efinance as ef  # efinance不能连国际VPN
#
# from seagull.settings import PATH
# from seagull.utils import utils_database, utils_log, utils_data
#
# with utils_database.engine_conn("POSTGRES") as conn:
#     minute = pd.read_sql("select distinct 股票代码 from ods_ohlc_incr_efinance_stock_minute", con=conn.engine)
#     daily = pd.read_sql("select distinct 股票代码 from ods_ohlc_incr_efinance_stock_daily", con=conn.engine)
# import pandas as pd
# from seagull.settings import PATH
# from seagull.utils import utils_database, utils_log, utils_data
# with utils_database.engine_conn("POSTGRES") as conn:
#     # self.dwd_stock_bj_base_df = pd.read_sql("select * from dwd_info_incr_stock_base where market_code='bj'", con=conn.engine)
#     dwd_stock_base_df = pd.read_sql(
#         "select * from dwd_info_incr_adata_stock_base", con=conn.engine)
#     minute = pd.read_sql(
#         "select distinct 股票代码 from ods_ohlc_incr_efinance_stock_minute", con=conn.engine)
#     dwd_stock_base_df1 = dwd_stock_base_df[~dwd_stock_base_df.asset_code.isin(minute['股票代码'])]
#
#
# Out[4]:
#      asset_code market_code  full_code     insert_timestamp
# 110      000508          sz  000508.sz  2025-05-27 09:52:43
# 1592     200539          sz  200539.sz  2025-05-27 09:52:43
# 1593     200541          sz  200541.sz  2025-05-27 09:52:43
# 1594     200550          sz  200550.sz  2025-05-27 09:52:43
# 1596     200570          sz  200570.sz  2025-05-27 09:52:43
#          ...         ...        ...                  ...
# 5609     920445          bj  920445.bj  2025-05-27 09:52:43
# 5610     920489          bj  920489.bj  2025-05-27 09:52:43
# 5611     920682          bj  920682.bj  2025-05-27 09:52:43
# 5612     920799          bj  920799.bj  2025-05-27 09:52:43
# 5613     920819          bj  920819.bj  2025-05-27 09:52:43
# [2577 rows x 4 columns]

