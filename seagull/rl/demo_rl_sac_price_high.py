# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:44:48 2024

@author: awei

(demo_rl_sac_price_high)
"""
import argparse
from datetime import datetime  # , timedelta
# import math
import os

import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
import pandas as pd

from __init__ import path
from trade import trade_eval
from base import base_connect_database, base_utils
from rl import rl_sac
from utils import utils_log  # utils_database, utils_data, 

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

RE_ENVIRONMENT = 'dwd_freq_incr_stock_daily'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SACAgentTrain(rl_sac.SACAgent):
    """
    Remark:
        sell_pct % in valume
        buy_pct % in capital
    """
    def __init__(self, state_dim, action_dim, all_stock):
        super().__init__(state_dim, action_dim)
        self.all_stock = all_stock
    def simulate_trading(self, data, date_start, date_end, dataset_type):
        insert_timestamp = datetime.now().strftime('%F %T')
        primary_key_episodes = base_utils.md5_str(insert_timestamp)
        
        
        date_sorted = sorted([x for x in data.date.unique() if str(x)!='nan'])
        if dataset_type=='train':
            episodes = 1000
        elif dataset_type=='test':
            episodes = 1
            date_sorted *= 2 
        elif dataset_type=='real':
            episodes = 1
            
        print(f'date_sorted: {date_sorted}')
        
        #data['positions_pct'] = np.random.rand(data.shape[0]) #init
        
        for episode in range(episodes):  # Simulate trading for 100 episodes
            if episode % 10 == 0:  # Check if the episode is a multiple of 10
                print(f'episode: {episode} / {episodes}  | {datetime.now().strftime("%F %T")}')
             
            trading_model_list, transaction_order_list = [], []
            # Initial
            capital_init = 1_000_000_000
            capital = capital_init
    agent = SACAgentTrain(state_dim, action_dim, all_stock)
    agent.simulate_trading(backtest_df, date_start, date_end, dataset_type='train')

def __apply_completion(state_subtable, all_stock):
    date = state_subtable.name
    #state_subtable
    merged_state = pd.merge(all_stock[['code', 'code_name']], state_subtable, on='code', how='left').fillna(0.0)
    merged_state['date'] = date
    
    # Fill NaN values with False
    #merged_state.fillna(False, inplace=True)
    
    # Sort the merged dataframe based on the order of 'all_stock'
    #sorted_state = merged_state.sort_values(by='code')
    
    #print(sorted_state)
    return merged_state

def train_process(date_start='2020-01-01', date_end='2022-01-01'):
    with base_connect_database.engine_conn('postgre') as conn:
        sql = f"SELECT * FROM {RE_ENVIRONMENT} WHERE date >= '{date_start}' AND date < '{date_end}'"#rl_environment
        backtest_raw_df = pd.read_sql(sql, con=conn.engine)
    primary_key
    date
    time
    board_type
    full_code
    asset_code
    market_code
    code_name
    price_limit_pct
    open
    high
    low
    close
    prev_close
    volume
    amount
    turn
    trade_status
    pct_chg
    pe_ttm
    ps_ttm
    pcf_ncf_ttm
    pb_mrq
    is_st
    insert_timestamp
    amplitude
    price_chg
    freq
    adj_type
    board_primary_key
    backtest_raw_df = backtest_raw_df.drop(columns=['code_name', 'insert_timestamp'])
    backtest_raw_df['total_held_volume_pct'] = 0.0
    backtest_raw_df = backtest_raw_df.drop_duplicates(['date','code'], keep='first') # 处理测试导致的重复数据污染
    #backtest_raw_df.index = backtest_raw_df['code'] + '_' + backtest_raw_df['date']
    #backtest_raw_df = backtest_raw_df.sort_values(by='date',ascending=True).reset_index(drop=True)
    
    with base_connect_database.engine_conn('postgre') as conn:
# =============================================================================
#         date_end = datetime.now()
#         date_start = date_end - timedelta(days=182)  # 距今半年有数据
#         history_a_stock_k_data = pd.read_sql(f"SELECT primary_key, date, code FROM history_a_stock_k_data where date>='{date_start.strftime('%F')}';", con=conn.engine)
# =============================================================================
        # history_a_stock_k_data.code.value_counts()
        all_stock = pd.read_sql("SELECT * FROM all_stock_copy", con=conn.engine)
        all_stock = all_stock[all_stock.tradeStatus=='1']
        #all_stock['held_volume'] = 0.0
        #all_stock.index = all_stock.code
        
    backtest_raw_df = backtest_raw_df.groupby('date').fillna(0.0)
    backtest_df = backtest_raw_df[['date','code'] + FEATURE]




    # code_name may change, so use the latest all_stock to map the corresponding code_name
    all_stock_dict = dict(zip(all_stock['code'], all_stock['code_name']))
    backtest_df['code_name'] = backtest_df['code'].map(all_stock_dict)

# =============================================================================
#     feature_df = backtest_df.loc[backtest_df.date==trading_day, ['rear_low_pct_pred', 'rear_high_pct_pred', 'high', 'low', 'close', 'preclose', 'code', 'held_volume']]
#     feature_df = pd.merge(all_stock, feature_df, on='code').reset_index(drop=True)
# =============================================================================
    
    
    #backtest_df = backtest_df.groupby('date').apply(lambda subtable: pd.merge(all_stock, subtable, on='code')).reset_index(drop=True)
    
    #global backtest_df1
    #backtest_df1 = backtest_df
    
    # Initialize agent and environment
    #continuous_action_dim = len(backtest_df.columns)
    #state_dim = continuous_action_dim + 1
    state_dim = len(FEATURE)  # 13
    print('state_dim', state_dim)
    action_dim = 8
    discrete_action_dim = all_stock.shape[0] # 5826
    agent = SACAgentTrain(state_dim, action_dim, all_stock)
    agent.simulate_trading(backtest_df, date_start, date_end, dataset_type='train')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='Start time for backtesting')
    #parser.add_argument('--date_start', type=str, default='2021-10-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2024-01-01', help='End time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2024-03-01', help='End time for backtesting')
    args = parser.parse_args()
    
    logger.info(f"""task: demo_rl_sac_price_high
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")
    train_process(date_start=args.date_start,
                  date_end=args.date_end)
