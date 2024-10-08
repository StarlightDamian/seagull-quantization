# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:24:14 2024

@author: awei
ETF溢价率(portfolio_premium_rate)
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

def calculate_etf_premium(etf_price, nav, freq='1D'):
    """
    计算ETF的溢价率
    
    参数:
    etf_price: pd.Series, ETF的市场价格
    nav: pd.Series, ETF的净值(NAV)
    freq: str, 重采样频率，默认为'1D'（每日）
    
    返回:
    vbt.Portfolio, 包含溢价率计算结果的Portfolio对象
    """
    # 确保日期索引对齐
    df = pd.concat([etf_price, nav], axis=1, keys=['price', 'nav'])
    df = df.resample(freq).last().dropna()
    
    # 计算溢价率
    premium = (df['price'] - df['nav']) / df['nav']
    
    # 创建vectorbt的Portfolio对象
    portfolio = vbt.Portfolio.from_holding(premium, freq=freq)
    
    return portfolio

# 示例使用
# 假设我们有ETF价格和NAV的日期序列数据
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
etf_price = pd.Series(np.random.uniform(100, 110, len(dates)), index=dates)
nav = pd.Series(np.random.uniform(99, 111, len(dates)), index=dates)

# 计算溢价率
premium_portfolio = calculate_etf_premium(etf_price, nav)

# 输出基本统计信息
print(premium_portfolio.total_return())
print(premium_portfolio.sharpe_ratio())

# 绘制溢价率随时间变化的图表
premium_portfolio.plot().show()

# 计算年化溢价率
annual_premium = premium_portfolio.total_return() * (252 / len(premium_portfolio.close))
print(f"年化溢价率: {annual_premium:.2%}")

# 找出最大溢价和折价
max_premium = premium_portfolio.max()
max_discount = premium_portfolio.min()
print(f"最大溢价: {max_premium:.2%}")
print(f"最大折价: {max_discount:.2%}")

# 计算溢价率的波动率
premium_volatility = premium_portfolio.close.std() * np.sqrt(252)  # 假设252个交易日
print(f"溢价率年化波动率: {premium_volatility:.2%}")