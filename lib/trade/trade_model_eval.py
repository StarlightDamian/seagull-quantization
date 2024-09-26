# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:14:42 2023

@author: awei
交易模型评估(trade_model_eval) #,test
多个模型和历史数据横向/纵向比较
"""
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from __init__ import path
from base import base_connect_database


TRADE_ORDER_TABLE_NAME = 'trade_order_details'
TRADE_MODEL_TABLE_NAME = 'trade_model'
# Set a higher DPI value for better resolution
plt.figure(figsize=(10, 6), dpi=600)

def annualized_income_max_primary_key():
    with base_connect_database.engine_conn('postgre') as conn:
        trade_model_df = pd.read_sql(f"SELECT * FROM {TRADE_MODEL_TABLE_NAME}", con=conn.engine)
    annualized_income_max = trade_model_df.annualized_income.max()
    primary_key = trade_model_df.loc[trade_model_df.annualized_income==annualized_income_max, 'primary_key_order'].values[0]
    print(f'annualized_income_max: {annualized_income_max} % |primary_key: {primary_key}')
    return primary_key

def rl_first_model_analyze(code):
    primary_key = annualized_income_max_primary_key(code)
    with base_connect_database.engine_conn('postgre') as conn:
        trade_order_details_df = pd.read_sql(f"SELECT * FROM {TRADE_ORDER_TABLE_NAME}", con=conn.engine)
    print(f'primary_key: {primary_key}')
    rl_first_model_analyze = trade_order_details_df[trade_order_details_df.primary_key==primary_key]
    rl_first_model_analyze.to_csv(f'{path}/data/rl_first_model_analyze.csv', index=False)

def trade_primary_key(code):
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='End time for backtesting')
    args = parser.parse_args()
    
    primary_key = annualized_income_max_primary_key(code)
    with base_connect_database.engine_conn('postgre') as conn:
        backtest_sql = f"SELECT * FROM prediction_stock_price_test WHERE date >= '{args.date_start}' AND date < '{args.date_end}' and code='sz.002230' "
        backtest_df = pd.read_sql(backtest_sql, con=conn.engine)
        
        trade_share_register_df = pd.read_sql(f"SELECT * FROM {TRADE_ORDER_TABLE_NAME}", con=conn.engine)
    
    trade_primary_key_df = trade_share_register_df[trade_share_register_df.primary_key==primary_key]
    trade_primary_key_df.to_csv(f"{path}/data/trade_primary_key_df.csv", index=False)

if __name__ == '__main__':
    with base_connect_database.engine_conn('postgre') as conn:
    #conn = base_connect_database.engine_conn('postgre')
        trade_model_df = pd.read_sql(f"SELECT * FROM trade_model", con=conn.engine)
        
    # Loop through unique primary_key values
    for primary_key in trade_model_df.primary_key.unique():
        # Filter the DataFrame for the current primary_key
        trade_model_1 = trade_model_df[trade_model_df.primary_key == primary_key]
        lr_actor, lr_critic = trade_model_1[['lr_actor', 'lr_critic']].values[0]
        # Use Seaborn to plot the line with different colors for each primary_key
        sns.lineplot(x='episode', y='valuation', data=trade_model_1, label=f'Primary Key {primary_key},lr_actor{lr_actor},lr_critic{lr_critic}')
    
    # Display legend to differentiate the lines
    plt.legend()
    
    # Display the plot
    plt.show()
# =============================================================================
#     for primary_key in trade_model_df.primary_key.unique():
#         trade_model_1 = trade_model_df[trade_model_df.primary_key==primary_key]
#         # 使用 seaborn 绘制折线图
#         sns.lineplot(x='episode', y='valuation', data=trade_model_1)
#         
#         # 显示图形
#         plt.show()
# =============================================================================
    