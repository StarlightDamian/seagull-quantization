# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:02:24 2023

@author: awei
强化学习测试曲线(testing_rl_eval)
"""
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from seagull.settings import PATH
from base import base_connect_database

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
TRADE_ORDER_TABLE_NAME = 'trade_order_details'
# Set a higher DPI value for better resolution

def backtesting_rl(code, code_name, date_start, date_end):
    plt.figure(figsize=(10, 6), dpi=600)
    # Data exploration
    with base_connect_database.engine_conn("POSTGRES") as conn:
        #data = pd.read_sql("SELECT * FROM history_a_stock_k_data limit 10", con=conn.engine)  # Use conn_pg.engine
        trade_order_details = pd.read_sql(f"SELECT * FROM {TRADE_ORDER_TABLE_NAME} WHERE dataset_type='test' ", con=conn.engine)
        print(trade_order_details)
        
        sql = f"SELECT * FROM history_a_stock_k_data WHERE date >= '{date_start}' AND date < '{date_end}' and code='{code}' "
        history_a_stock_k_data = pd.read_sql(sql, con=conn.engine)
    

    
    trade_order_deal_details = trade_order_details[(trade_order_details.code==code)&(trade_order_details.state=='make_a_deal')]
    trade_order_deal_details = trade_order_deal_details.sort_values(by='date',ascending=True).reset_index(drop=True)
    #trade_order_deal_details = trade_order_deal_details.sort_values(by='date',ascending=True).reset_index(drop=True)
    

    # reference line
    close_start = history_a_stock_k_data.close.values[0]
    history_a_stock_k_data['volume_raw'] = 1_000_000 / close_start
    history_a_stock_k_data['valuation_raw'] = history_a_stock_k_data.volume_raw * history_a_stock_k_data.close
    sns.lineplot(x='date', y='valuation_raw', data=history_a_stock_k_data, label=f'{code} {code_name}', color='black')
    # history_a_stock_k_data[['date', 'valuation_raw']]
    
    
    # Loop through unique primary_key values
    for primary_key in trade_order_deal_details.primary_key.unique():
        # Filter the DataFrame for the current primary_key
        trade_model_1 = trade_order_deal_details[trade_order_deal_details.primary_key == primary_key]
        # trade_model_1[['date', 'valuation']]
        
        #lr_actor, lr_critic = trade_model_1[['lr_actor', 'lr_critic']].values[0]
        # Use Seaborn to plot the line with different colors for each primary_key
        sns.lineplot(x='date', y='valuation', data=trade_model_1, label='模型回测收益曲线')#Primary Key {primary_key}
    
    # Display legend to differentiate the lines
    
    plt.legend()
    plt.title(f'回测日期: {date_start} _ {date_end}')
    plt.ylabel('估值')
    # Display the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2022-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    #parser.add_argument('--date_start', type=str, default='2023-06-01', help='Start time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2023-12-23', help='End time for backtesting')
    args = parser.parse_args()
    
    stock_dict = {'sz.002281': '光迅科技',
                  'sz.002747': '埃斯顿',
                  'sh.600522': '中天科技',
                  'sh.600732': '爱旭股份',
                  }

    for (code, code_name) in stock_dict.items():
        backtesting_rl(code=code, code_name=code_name, date_start=args.date_start, date_end=args.date_end)
    
    

    