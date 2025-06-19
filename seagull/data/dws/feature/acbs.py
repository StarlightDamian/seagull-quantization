# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:12:43 2024

@author: awei

持仓成本Average Cost Basis(acbs)
"""

import numpy as np
import pandas as pd


def single_stock_cost_distribution(group, num_bins):
    prices = group['close'].values
    volumes = group['volume'].values
    
    # 计算累积成交量
    cum_volume = np.cumsum(volumes)
    total_volume = cum_volume[-1]
    
    # 计算加权平均价格（即理论上的平均持仓成本）
    vwap = np.sum(prices * volumes) / total_volume
    
    # 创建价格区间
    price_range = np.linspace(prices.min() * 0.9, prices.max() * 1.1, num_bins + 1)
    
    # 计算每个价格区间的成交量
    volume_dist = np.zeros(num_bins)
    for i in range(len(prices)):
        idx = np.digitize(prices[i], price_range) - 1
        volume_dist[idx] += volumes[i]
    
    # 归一化分布
    volume_dist /= total_volume
    
    return pd.Series({
        'vwap': vwap,
        'price_min': price_range[0],
        'price_max': price_range[-1],
        'distribution': volume_dist.tolist()
    })

def calculate_market_position_cost_distribution(data, window_size=30, num_bins=100):
    """
    计算市场范围内的持仓成本分布
    
    参数:
    data: DataFrame, 多层索引 (股票代码, 日期), 列包括 'close' (收盘价) 和 'volume' (成交量)
    window_size: int, 计算窗口大小（交易日）
    num_bins: int, 用于计算分布的区间数量
    
    返回:
    DataFrame, 包含每只股票在每个时间点的持仓成本分布
    """
    # 对每只股票在滚动窗口内计算持仓成本分布
    result = data.groupby(level=0).rolling(window=window_size).apply(single_stock_cost_distribution, num_bins)
    return result

# 示例使用
# 假设我们有一个包含多只股票数据的DataFrame
data = pd.DataFrame({
    'close': np.random.uniform(10, 100, 1000),
    'volume': np.random.randint(1000, 10000, 1000)
}, index=pd.MultiIndex.from_product([
    ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'],  # 股票代码
    pd.date_range('2023-01-01', periods=200)  # 日期
]))

result = calculate_market_position_cost_distribution(data)
print(result)