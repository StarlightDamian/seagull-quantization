# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:52:19 2023

@author: awei
均值回归策略
Mean Reversion Strategy
尝试利用股票价格波动的短期波动性。策略可以根据历史价格数据来识别价格偏离均值的机会，并在价格回归到均值水平时进行交易。
"""
import numpy as np
import pandas as pd

# 模拟价格数据（示例数据）
np.random.seed(0)
price_data = np.random.normal(100, 10, 252)  # 252个交易日的价格数据，均值100，标准差10

# 定义均值回归策略函数
def mean_reversion_strategy(balance, price_data):
    initial_balance = balance
    num_periods = len(price_data)
    position = 0  # 初始持仓为0

    for t in range(1, num_periods):
        if price_data[t] > np.mean(price_data[:t]) and position < 0:
            # 如果价格高于均值且持仓为负，则买入
            balance += price_data[t] * abs(position)
            position = 0
        elif price_data[t] < np.mean(price_data[:t]) and position == 0:
            # 如果价格低于均值且没有持仓，则卖出
            position = int(balance / price_data[t])
            balance -= price_data[t] * position

    # 最后一天平仓
    balance += price_data[-1] * position
    position = 0

    return balance, num_periods

# 初始化参数
initial_balance = 1_000_000
num_simulations = 1_000
results = []

# 进行1000次模拟
for i in range(num_simulations):
    final_balance, num_periods = mean_reversion_strategy(initial_balance, price_data)
    results.append([final_balance, num_periods])

# 将结果转换为DataFrame
df = pd.DataFrame(results, columns=['final_balance', 'periods'])

# 计算平均值
average_final_balance = df['final_balance'].mean()
average_num_periods = df['periods'].mean()

# 打印平均结果
print("平均最终资金余额:", average_final_balance)
print("平均周期:", average_num_periods)

# 可选：将结果保存为CSV文件
df.to_csv('mean_reversion_results.csv', index=False)

