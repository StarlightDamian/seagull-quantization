# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:53:43 2023

@author: awei
尝试构建一些特征(data_feature)
"""
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from seagull.settings import PATH
from base import base_connect_database

def apply_amount(subtable):
    return subtable.amount.sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='End time for backtesting')
    args = parser.parse_args()
    
    with base_connect_database.engine_conn("POSTGRES") as conn:
        history_sql = f"SELECT * FROM history_a_stock_k_data WHERE date >= '{args.date_start}' AND date < '{args.date_end}' and code='sz.002230' "
        history_df = pd.read_sql(history_sql, con=conn.engine)
        
        
    
    amount_sum_series = history_df.groupby('date').apply(apply_amount)
    
    amount_sum_series.index = pd.to_datetime(amount_sum_series.index)
    
    # Plotting the Series
    amount_sum_series.plot(kind='line', marker='o', linestyle='-')
    
    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Amount Sum')
    plt.title('Line Chart of Amount Sum')
    
    # Display the plot
    plt.show()