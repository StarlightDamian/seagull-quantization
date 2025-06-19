# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:52:54 2024

@author: awei
vap9
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 假设我们有一个包含股票交易数据的DataFrame
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'open': [100, 102, 98, 105, 103],
    'high': [101, 103, 100, 108, 104],
    'low': [99, 101, 97, 103, 102],
    'close': [100.5, 102.5, 92.5, 106.5, 103.5],
    'volume': [10000, 12000, 8000, 15000, 11000]
})

# 计算筹码分布
price_ranges = [90, 95, 100, 105, 110, 115]
chip_distribution = pd.DataFrame(columns=['price_range', 'volume'])

for i in range(len(price_ranges) - 1):
    low = price_ranges[i]
    high = price_ranges[i + 1]
    volume = df.loc[(df['low'] >= low) & (df['high'] < high), 'volume'].sum()
    chip_distribution = chip_distribution._append({'price_range': f"{low} - {high}", 'volume': volume}, ignore_index=True)

# 显示筹码分布
print(chip_distribution)

# 绘制筹码分布图
plt.figure(figsize=(12, 6))
sns.barplot(x='price_range', y='volume', data=chip_distribution)
plt.title('Stock Chip Distribution')
plt.xlabel('Price Range')
plt.ylabel('Volume')
plt.xticks(rotation=45)
plt.show()
