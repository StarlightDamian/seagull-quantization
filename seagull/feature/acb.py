# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:29:02 2024

@author: awei

持仓成本Average Cost Basis(ACB)

持仓成本通常指的是买入某一资产时的平均价格，考虑到所有买入的交易。如果涉及多次买入和卖出，
且持有多个头寸，那么计算会稍微复杂一些，特别是需要考虑卖出对平均成本的影响。
"""

import pandas as pd
import numpy as np

def calculate_position_cost(trades):
    """
    计算持仓成本
    
    参数:
    trades: DataFrame, 包含列: 'datetime', 'price', 'size'
    
    返回:
    DataFrame, 包含列: 'datetime', 'price', 'size', 'position', 'position_cost'
    """
    
    # 确保交易按时间排序
    trades = trades.sort_values('datetime')
    
    # 初始化结果DataFrame
    result = trades.copy()
    result['position'] = result['size'].cumsum()
    result['position_cost'] = 0.0
    
    current_position = 0
    current_cost = 0
    
    for i, row in result.iterrows():
        if current_position == 0:
            # 如果之前没有持仓，新的持仓成本就是当前交易价格
            current_cost = row['price']
        elif (current_position > 0 and row['size'] > 0) or (current_position < 0 and row['size'] < 0):
            # 如果增加已有的多头或空头仓位
            current_cost = (current_position * current_cost + row['size'] * row['price']) / (current_position + row['size'])
        elif (current_position > 0 and row['size'] < 0) or (current_position < 0 and row['size'] > 0):
            # 如果减少已有的仓位或反向操作，成本不变
            pass
        
        current_position += row['size']
        result.at[i, 'position_cost'] = current_cost if current_position != 0 else 0
    
    return result

# 示例使用
trades = pd.DataFrame({
    'datetime': pd.date_range(start='2023-01-01', periods=5),
    'price': [100, 102, 98, 103, 101],
    'size': [10, 5, -3, 8, -7]
})

result = calculate_position_cost(trades)
print(result)