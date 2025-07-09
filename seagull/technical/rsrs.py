# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:28:13 2024

@author: awei
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_rsrs(data, window=14):
    """
    计算 RSRS 相对支撑位阻力位指标
    :param data: DataFrame，必须包含 'high' 和 'low' 列
    :param window: 滑动窗口大小
    :return: DataFrame，新增 'beta' 和 'zscore' 列
    """
    betas = []
    
    for i in range(len(data)):
        if i < window - 1:
            betas.append(np.nan)
        else:
            window_data = data.iloc[i-window+1:i+1]
            slope, intercept, r_value, p_value, std_err = linregress(window_data['low'], window_data['high'])
            betas.append(slope)
    
    data['beta'] = betas
    data['zscore'] = (data['beta'] - np.nanmean(betas)) / np.nanstd(betas)
    
    return data

if __name__ == '__main__':

    # 示例数据
    data = pd.DataFrame({
        'high': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'low': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    })
    
    result = calculate_rsrs(data, window=5)
    print(result[['beta', 'zscore']])



import pandas as pd
import talib

# 假设 df 是你的输入数据，包含了'close', 'high', 'low', 'full_code', 'date'等列

def compute_rsrs(group):
    # 计算 RSI（相对强弱指数）
    group['RSI'] = talib.RSI(group['close'], timeperiod=14)
    
    # 计算移动平均线（例如 50 日和 200 日）
    group['MA50'] = talib.SMA(group['close'], timeperiod=50)
    group['MA200'] = talib.SMA(group['close'], timeperiod=200)
    
    # 计算收盘价相对于50日均线的强度（示例）
    group['RSRS'] = (group['close'] - group['MA50']) / group['MA50']  # 收盘价与50日均线的差值
    
    return group

# 假设 df 是原始数据
df['date'] = pd.to_datetime(df['date'])  # 确保日期列是datetime类型

# 使用 groupby 进行分组，并应用计算函数
result = df.groupby('full_code').apply(compute_rsrs)

# 输出结果
print(result[['full_code', 'date', 'RSI', 'MA50', 'MA200', 'RSRS']].head())
