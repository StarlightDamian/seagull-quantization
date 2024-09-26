# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:04:15 2023

@author: awei
"""
import sys
import os
#import os

# 获取当前脚本所在目录的路径
current_directory = os.path.dirname(os.path.abspath(__file__))

print("当前脚本所在目录的路径:", current_directory)

# 获取当前工作目录（当前文件夹）的路径
#current_directory = os.getcwd()
# 构建上两级目录的路径
parent_directory = os.path.dirname(current_directory)  # 上一级目录
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

path = os.path.dirname(parent_directory)  # 上两级目录
if path not in sys.path:
    sys.path.append(path)

#from base import base_arguments
#from base import base_utils

if __name__ == '__main__':
    print(path)
    
    
    
