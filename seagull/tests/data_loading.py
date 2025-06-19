# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:40:07 2023

@author: awei
数据处理
"""
import os
import argparse

import pandas as pd

from seagull.settings import PATH
from base import base_utils


def re_get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result

def feather_file_merge(date_start, date_end):
    date_binary_pair_list = base_utils.date_binary_list(date_start, date_end)
    feather_files = [f'{PATH}/_file/day/{date_binary_pair[0]}.feather' for date_binary_pair in date_binary_pair_list]
    #print(feather_files)
    dfs = [pd.read_feather(file) for file in feather_files if os.path.exists(file)]
    feather_df = pd.concat(dfs, ignore_index=True)
    return feather_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-02-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    # 获取日期段数据
    date_range_df = feather_file_merge(args.date_start, args.date_end)
    print(date_range_df)
    