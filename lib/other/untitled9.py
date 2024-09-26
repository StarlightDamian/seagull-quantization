# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 19:01:21 2023

@author: awei
"""
# =============================================================================
# import pandas as pd
# from tabulate import tabulate
# 
# # 创建一个示例 DataFrame
# data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#         'Age': [25, 30, 35, 28],
#         'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
# df = pd.DataFrame(data)
# 
# # 将 DataFrame 转换为纯文本表格
# table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
# 
# # 打印表格
# print(table)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3), dpi=600)
# 创建一个示例 DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)

# 使用 matplotlib 绘制表格
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')  # 不显示坐标轴

# 使用 table 方法将 DataFrame 转换为表格，并添加到图表中
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='upper left')

# 调整表格布局
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.show()

