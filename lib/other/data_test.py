# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:36:16 2024

@author: awei

"""
import pandas as pd
from __init__ import path
from base import base_connect_database, base_utils
import baostock as bs
bs.login()
#with base_connect_database.engine_conn('postgre') as conn:
#    stock_df = pd.read_sql("dwd_a_stock_day", con=conn.engine)
df = bs.query_history_k_data_plus('sh.600176',fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST')
bs.logout()


# 去掉前面的“sh.”、“sz.”、“bj.”
df1['code'] = df1['code'].str.replace(r'^[a-z]{2}\.', '', regex=True)

with base_connect_database.engine_conn('postgre') as conn:
    max_date = pd.read_sql("SELECT max(date) FROM ods_a_stock_day", con=conn.engine)
    
# =============================================================================
# def download_data(date):
#     bs.login()
# 
#     # 获取指定日期的指数、股票数据
#     stock_rs = bs.query_all_stock(date)
#     stock_df = stock_rs.get_data()
#     data_df = pd.DataFrame()
#     for code in stock_df["code"]:
#         print("Downloading :" + code)
#         k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
#         data_df = data_df.append(k_rs.get_data())
#     bs.logout()
#     data_df.to_csv("E:/download/demo_assignDayData.csv", encoding="gbk", index=False)
#     print(data_df)
# 
# 
# if __name__ == '__main__':
#     # 获取指定日期全部股票的日K线数据
#     download_data("2019-02-25")
# 
# =============================================================================
#日线 =============================================================================
# import baostock as bs
# bs.login()
# k_rs = bs.query_history_k_data_plus('sh.000001', "date,code,open,high,low,close", "2023-12-26", "2023-12-26")
# bs.logout()
# print(k_rs.get_data())
# =============================================================================

# =============================================================================
# #5分钟线
# import baostock as bs
# bs.login()
# k_rs = bs.query_history_k_data_plus('sh.000001',
#                                     "date,code,open,high,low,close",
#                                     "2023-12-26",
#                                     "2023-12-26",
#                                     frequency='5')
# bs.logout()
# print(k_rs.get_data())
# =============================================================================
