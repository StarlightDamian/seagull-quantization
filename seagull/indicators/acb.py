# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:23:47 2024

@author: awei
持仓成本Average Cost Basis(ACB)

持仓成本通常指的是买入某一资产时的平均价格，考虑到所有买入的交易。如果涉及多次买入和卖出，
且持有多个头寸，那么计算会稍微复杂一些，特别是需要考虑卖出对平均成本的影响。
"""

import pandas as pd  
  
# 示例交易数据：日期、操作类型（buy/sell）、数量和价格  
transactions = [  
    {'date': '2023-01-01', 'type': 'buy', 'quantity': 10, 'price': 100},  
    {'date': '2023-01-02', 'type': 'buy', 'quantity': 5, 'price': 110},  
    {'date': '2023-01-03', 'type': 'sell', 'quantity': 3, 'price': 120},  
    {'date': '2023-01-04', 'type': 'buy', 'quantity': 7, 'price': 130},  
]  
  
# 将交易数据转换为DataFrame  
df_transactions = pd.DataFrame(transactions)  
df_transactions['date'] = pd.to_datetime(df_transactions['date'])  
  
# 计算持仓成本和持仓数量  
df_transactions['cumulative_buy_quantity'] = df_transactions[df_transactions['type'] == 'buy']['quantity'].cumsum()  
df_transactions['cumulative_buy_value'] = df_transactions[df_transactions['type'] == 'buy']['quantity'] * df_transactions[df_transactions['type'] == 'buy']['price'].cumsum() / df_transactions['cumulative_buy_quantity'].shift(1, fill_value=1)  
df_transactions['average_cost'] = df_transactions['cumulative_buy_value'].where(df_transactions['type'] == 'buy', df_transactions['cumulative_buy_value'].shift(fill_value=0))  
  
# 处理卖出操作对持仓成本的影响（这里假设卖出不影响未卖出部分的成本）  
# 注意：这只是一个简化的例子，实际情况可能需要更复杂的逻辑来处理卖出后的成本调整  
sell_indices = df_transactions[df_transactions['type'] == 'sell'].index  
for idx in sell_indices:  
    prev_idx = df_transactions.index.get_loc(idx) - 1  
    if prev_idx >= 0:  
        df_transactions.loc[idx, 'average_cost'] = df_transactions.loc[prev_idx, 'average_cost']  
  
# 仅保留最新持仓成本的记录（假设我们只需要最终的持仓成本）  
latest_cost = df_transactions.iloc[-1]['average_cost'] if df_transactions.iloc[-1]['type'] != 'sell' else df_transactions.iloc[df_transactions[df_transactions['type'] != 'sell'].index[-1]]['average_cost']  
  
print("最终的持仓成本:", latest_cost)