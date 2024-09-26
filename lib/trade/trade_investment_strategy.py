# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:46:09 2023

@author: awei
投资策略(trade_investment_strategy)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database

HISTORICAL_PRICE_TABLE_NAME = 'prediction_stock_price_test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='Starting time for simulated trading')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='End time for simulated trading')
    args = parser.parse_args()

    print(f'Starting time for simulated trading: {args.date_start}\nEnd time for simulated trading: {args.date_end}')
    
    
    with base_connect_database.engine_conn('postgre') as conn:
        historical_price_df = pd.read_sql(f"SELECT * FROM {HISTORICAL_PRICE_TABLE_NAME} WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(historical_price_df)
    
    trade_price_code = historical_price_df[historical_price_df.code=='sz.002230']
    trade_price_code.to_csv(f'{path}/data/trade_price_code.csv', index=False)
    
# =============================================================================
#     print('Trading days:',len(historical_price_df.date.unique()))
#     
#     date = '2023-03-01'
#     day_price_df = historical_price_df[historical_price_df.date == '2023-03-01']
#     
#     #day_price_df['rearDiffPctChgPred'] = day_price_df.rearHighPctChgPred - day_price_df.rearLowPctChgPred
#     #day_price_sort_df = day_price_df.sort_values(by='rearDiffPctChgPred', ascending=False)
#     #day_price_sort_df = day_price_df.sort_values(by='rearHighPctChgPred', ascending=False)
#     day_price_sort_df = day_price_df.sort_values(by='rearDiffPctChgPred', ascending=False)
#     
#     day_price_sort_df = day_price_sort_df[~(day_price_sort_df.remarks=='limit_up')]#.reset_index(drop=True) 序号
#     day_price_sort_df.head(10).to_csv(f'{path}/data/trade_price_day.csv', index=False)
# =============================================================================
    
    
    