# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:04:15 2023

@author: awei
"""
import sys
import os

# from base import base_log

# 获取当前脚本所在目录的路径
current_directory = os.path.dirname(os.path.abspath(__file__)) # debug: os.getcwd()在linux会显示根路径

# 构建上两级目录的路径
parent_directory = os.path.dirname(current_directory)  # 上一级目录
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

path = os.path.dirname(parent_directory)  # 上两级目录
if path not in sys.path:
    sys.path.append(path)

# logger = base_log.logger_config_base()
# logger.info(f'The path where the current script is located:\n{current_directory}')
print(f'The path where the current script is located:\n{current_directory}')

if __name__ == '__main__':
    print(path)
    
    
    
