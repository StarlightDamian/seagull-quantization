# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:06:56 2023

@author: awei
大盘数据特征工程
"""
import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取公司业绩快报 ####
rs = bs.query_performance_express_report("sh.600000", start_date="2015-01-01", end_date="2024-12-31")
print('query_performance_express_report respond error_code:'+rs.error_code)
print('query_performance_express_report respond  error_msg:'+rs.error_msg)

result_list = []
while (rs.error_code == '0') & rs.next():
    result_list.append(rs.get_row_data())
    # 获取一条记录，将记录合并在一起
result = pd.DataFrame(result_list, columns=rs.fields)
#### 结果集输出到csv文件 ####
#result.to_csv("D:\\performance_express_report.csv", encoding="gbk", index=False)
print(result)

#### 登出系统 ####
bs.logout()
# =============================================================================
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取行业分类数据
# rs = bs.query_stock_industry()
# # rs = bs.query_stock_basic(code_name="浦发银行")
# print('query_stock_industry error_code:'+rs.error_code)
# print('query_stock_industry respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# industry_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     industry_list.append(rs.get_row_data())
# result = pd.DataFrame(industry_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/stock_industry.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# 
# =============================================================================
# =============================================================================
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取货币供应量(年底余额)
# rs = bs.query_money_supply_data_year(start_date="2010", end_date="2035")
# print('query_money_supply_data_year respond error_code:'+rs.error_code)
# print('query_money_supply_data_year respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/money_supply_data_year.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# =============================================================================
# =============================================================================
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取货币供应量
# rs = bs.query_money_supply_data_month(start_date="2010-01", end_date="2025-12")
# print('query_money_supply_data_month respond error_code:'+rs.error_code)
# print('query_money_supply_data_month respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/money_supply_data_month.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# =============================================================================
# =============================================================================
# #存款准备金率：query_required_reserve_ratio_data()
# 
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取存款准备金率
# rs = bs.query_required_reserve_ratio_data(start_date="2010-01-01", end_date="2025-12-31")
# print('query_required_reserve_ratio_data respond error_code:'+rs.error_code)
# print('query_required_reserve_ratio_data respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/required_reserve_ratio.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# =============================================================================
# =============================================================================
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取贷款利率
# rs = bs.query_loan_rate_data(start_date="2010-01-01", end_date="2025-12-31")
# print('query_loan_rate_data respond error_code:'+rs.error_code)
# print('query_loan_rate_data respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/loan_rate.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# =============================================================================
# =============================================================================
# import baostock as bs
# import pandas as pd
# 
# # 登陆系统
# lg = bs.login()
# # 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)
# 
# # 获取存款利率
# rs = bs.query_deposit_rate_data(start_date="2020-01-01", end_date="2023-12-01")
# print('query_deposit_rate_data respond error_code:'+rs.error_code)
# print('query_deposit_rate_data respond  error_msg:'+rs.error_msg)
# 
# # 打印结果集
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)
# # 结果集输出到csv文件
# #result.to_csv("D:/deposit_rate.csv", encoding="gbk", index=False)
# print(result)
# 
# # 登出系统
# bs.logout()
# =============================================================================


# =============================================================================
# import argparse
# 
# import pandas as pd
# 
# from __init__ import path
# from base import base_connect_database
# 
# HISTORY_TABLE_NAME = 'history_a_stock_k_data'
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--date_start', type=str, default='2022-01-01', help='Start time for backtesting')
#     parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
#     args = parser.parse_args()
# 
#     # Data exploration
#     with base_connect_database.engine_conn('postgre') as conn:
#         sql = f"""SELECT date, code, amount
#                 FROM {HISTORY_TABLE_NAME}
#                 WHERE code IN ('sh.000001', 'sz.399101', 'sz.399102', 'sz.399106');
#                 """
#         history_a_stock_k_data = pd.read_sql(sql, con=conn.engine)
# =============================================================================
        

# =============================================================================
# macro_sh000001
# macro_sz399101
# macro_sz399102
# macro_sz399106
# midstream
# micro_monthly_line
# micro_yearly_line
# 
# 
# 
# 
# =============================================================================
