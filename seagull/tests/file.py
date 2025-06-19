# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:46:02 2024

@author: awei
"""

import os
from seagull.settings import PATH

directory_path = f'{PATH}/_file/history_a_stock_5_min/2022-01-01_2022-10-10/'

# 获取指定目录下的所有文件
all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

print(f'All files in {directory_path}:')
for file_name in all_files:
    print(file_name)
