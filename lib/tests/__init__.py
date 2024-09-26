# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:40:14 2023

@author: awei
"""
import sys
import os

# 获取当前工作目录（当前文件夹）的路径
current_directory = os.getcwd()
# 构建上两级目录的路径
parent_directory = os.path.dirname(current_directory)  # 上一级目录
path = os.path.dirname(parent_directory)  # 上两级目录
if parent_directory not in sys.path:
    sys.path.append(parent_directory)
    
#print(sys.path)
#from test import test_main

if __name__ == '__main__':
    print(sys.path)