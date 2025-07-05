# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:06:10 2024

@author: awei
baostock的api数据处理工具(utils_api_baostock)
"""
import pandas as pd

def split_baostock_code(df):
    # 去掉前面的“sh.”、“sz.”、“bj.”
    # data_raw_df['asset_code'] = data_raw_df['code'].str.replace(r'^[a-z]{2}\.', '', regex=True)
    
    # 提取代码部分
    df['market_code'] = df['code'].str.split('.').str[0].str.lower()
    df['asset_code'] = df['code'].str.split('.').str[1]
    df['full_code'] = df['asset_code'] + '.' + df['market_code']
    
    # 输出结果
    return df

def get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result


if __name__ == '__main__':
    df = pd.DataFrame([['bj.430017', 1],['sh.430017', 1],['sz.002906',1]], columns=['code', 'trade_status'])
    result_df = split_baostock_code(df)
    