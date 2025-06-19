# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:37:44 2024

@author: awei
adv

average daily dollar volume for the past d days
"""

import pandas as pd

# 示例数据
data = {
    'date': ['2023-12-01', '2023-12-02', '2023-12-03', '2023-12-04', '2023-12-05'],
    'volume': [100000, 150000, 120000, 180000, 200000],
    'close': [10, 9, 8, 11, 12]
}

# 创建 DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# 计算每日的成交额
df['daily_dollar_volume'] = df['volume'] * df['close']

# 设置天数 (例如过去 5 天)
d = 5

# 计算过去 d 天的平均日成交额 (adv{d})
df['adv_5'] = df['daily_dollar_volume'].rolling(window=d).mean()

# 查看结果
print(df[['date', 'daily_dollar_volume', 'adv_5']])
