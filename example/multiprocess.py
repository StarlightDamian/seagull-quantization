# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:37:29 2024

@author: awei
"""

# =============================================================================
# from multiprocessing import Pool#, cpu_count
# 
# def compute_strategy(num):
#     print(num)
#     return results
# 
# args_list=[1,2,3,4,5,6,7,8]
# with Pool(processes=4) as pool:
#     results = pool.map(compute_strategy, args_list)
# 
# print(results)
#         
# =============================================================================
import multiprocessing
import os

def compute_strategy(num):
    # print(f"Process {os.getpid()}: Computing for {num}")
    # 模拟一些计算
    result = num * num
    return result

if __name__ == '__main__':
    args_list = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # 使用 'spawn' 方法来启动进程,这在Windows和Linux上都有效
    multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(compute_strategy, args_list)
    
    print("Final results:", results)