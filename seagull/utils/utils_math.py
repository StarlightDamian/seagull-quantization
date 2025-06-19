# -*- coding: utf-8 -*-
"""
@Date: 2024/11/19 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_math.py
@Description: 数学公式
"""
import math 
import numpy as np
import pandas as pd


def log_e(series):
    # 确保输入是 NumPy 数组
    series = series.astype(float)
    x = series.values
    
    # 创建输出数组
    result = np.zeros_like(x, dtype=float)  # 初始化结果数组

    # 处理大于1的情况: log(x)
    mask_greater_than_1 = x > 1
    result[mask_greater_than_1] = np.log(x[mask_greater_than_1])

    # 处理小于-1的情况: -log(-x)
    mask_less_than_minus_1 = x < -1
    result[mask_less_than_minus_1] = -np.log(-x[mask_less_than_minus_1])

    # 处理在[-1, 1]范围内的情况: 结果为0，已初始化为0

    # 处理空值
    result[np.isnan(x)] = np.nan

    return pd.Series(result, index=series.index)  # 转回为 pd.Series


def signed_log10(arr):
    # 正数log10，负数先取绝对值，再log10，再 * (-1)
    return np.where(arr > 0, np.log10(arr), -np.log10(np.abs(arr)))

# def log_e(x): # 单个值
#     if isinstance(x, pd.Series):
#         x=x.astype(float)
#     if x == 0:
#         return 0  # 避免 log(0) 报错
#     elif x > 0:
#         return math.log(x)  # 正数取 log(x)
#     else:
#         return -math.log(-x)  # 负数取 log(|x|) 再乘以 -1