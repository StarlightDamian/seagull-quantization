# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:15:49 2023

@author: awei
SAC算法_模型结构_训练过程(rl_sac_train)
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

from seagull.settings import PATH
from trade import trade_eval
from base import base_connect_database, base_utils
from rl import rl_sac
from seagull.utils import utils_log  # utils_database, utils_data,

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRADE_MODEL_TABLE_NAME = 'trade_model'
TRADE_ORDER_TABLE_NAME = 'trade_order_details'
TRADE_SHARES_HELD_TABLE_NAME = 'trade_shares_held_pct'
RE_ENVIRONMENT = 'rl_environment'  # 'test_stock_pick'  # rl_environment


FEATURE = ['rear_low_pct_pred',
           'rear_high_pct_pred',
           'rear_diff_pct_pred',
           'rear_open_pct_pred',
           'rear_close_pct_pred',
           'rear_next_rise_pct_pred',
           'rear_price_limit_pred',
           'preclose',
           'open',
           'high',
           'low',
           'close',
           'volume',
           'amount',
           'turn',
           'macro_amount',
           'macro_amount_diff_1',
           'pctChg',
           'total_held_volume_pct']

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
        """
        data.columns=Index(['date', 'code', 'rear_low_pct_pred', 'rear_high_pct_pred',
               'rear_diff_pct_pred', 'preclose', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg', 'total_held_volume_pct',
               'held_volume', 'code_name'],
              dtype='object')

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        data_raw : TYPE
            DESCRIPTION.
        date_start : TYPE
            DESCRIPTION.
        date_end : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        #all_stock['held_volume'] = 0
        #all_stock.index = all_stock.code
        #held_volume_df = all_stock[['held_volume']]
        
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
            #data.loc[0, ['total_shares_held_pct']] = total_shares_held_pct
            #held_volume_df['held_volume'] = 0
            #all_stock['held_volume'] = 0.0
            self.all_stock['held_volume'] = 0.0
            
            date_start = date_sorted[0]
            state = data.loc[data.date==date_start, FEATURE].values
            #print('1',state.shape)
            insert_timestamp_order = datetime.now().strftime('%F %T')
            primary_key_order = base_utils.md5_str(insert_timestamp_order)
            trading_model_dict = {'primary_key': primary_key_episodes,
                                  'primary_key_order': primary_key_order,
                                  'dataset_type': dataset_type,
                                  'episode': episode,
                                  'capital_init': capital_init,
                                  'lr_actor': self.lr_actor,
                                  'lr_critic': self.lr_critic,
                                  'date_start': date_start,
                                  'date_end': date_end,
                                  'insert_timestamp': insert_timestamp,
                                  }
            
            for trading_day in date_sorted[1:]:
                if dataset_type=='train':
                    action = self.get_action(state,dataset_type=dataset_type)
                if dataset_type=='test':
                    action = self.get_action(state, dataset_type=dataset_type)
                    
                #print('action', action)
                action_columns  = ['buy_price_pct', 'buy_trading_signals', 'buy_probability', 'buy_positions_pct',
                                   'sell_price_pct', 'sell_trading_signals', 'sell_probability', 'sell_positions_pct']  # 'dynamic_positions_pct'
                
                action_df = pd.DataFrame(action, columns=action_columns)



                
                # Because sigmoid reduces the confidence interval to [0,1] and enlarges it to [0,2]
                #action_df['buy_price_pct'] *= 2
                #action_df['sell_price_pct'] *= 2
                
                feature_df = data.loc[data.date==trading_day, ['rear_low_pct_pred', 'rear_high_pct_pred','rear_next_rise_pct_pred', 'high', 'low', 'close', 'preclose', 'code', 'code_name']].reset_index(drop=True)  #, 'held_volume'

                feature_df = pd.merge(feature_df, self.all_stock[['code', 'held_volume']], on='code')
                feature_action_df = pd.concat([feature_df, action_df], axis=1)
                #feature_action_df.to_csv(f'{PATH}/data/feature_action_df_test3.csv')
                
                #feature_action_df = feature_action_df[~(feature_action_df.close==0.0)]  # Exclude non-transaction data
                #positions_pct = feature_action_df.positions_pct.mean()  #Dynamic position ratio (close to the dynamic position position, more inclined to sell at a higher price, buy at a lower price, place more orders, trade less, control positions)ll
                
                
                #print(feature_action_df[['held_volume','close']])
                ## valuation & held_volume_pct
                feature_action_df['held_volume_valuation'] = feature_action_df['held_volume'] * feature_action_df['close']
                held_volume_valuation = feature_action_df['held_volume_valuation'].sum()
                valuation = held_volume_valuation + capital
                held_volume_pct = held_volume_valuation / valuation
                #print(f'{trading_day}: capital: {capital:.2f} |held_volume_pct: {held_volume_pct}')
                
                ## sell
                feature_action_df.loc[feature_action_df['sell_trading_signals']<0.5, 'sell_probability'] = 0.0
                feature_action_df['sell_probability'] = feature_action_df['sell_probability'] / feature_action_df['sell_probability'].sum()
                
                sell_positions_pct = feature_action_df['sell_positions_pct'].mean()
                sell_capital_total = valuation * sell_positions_pct
                feature_action_df['sell_capital'] = feature_action_df['sell_probability'] * sell_capital_total
                feature_action_df['sell_price'] = (1 + feature_action_df['rear_high_pct_pred'] *  feature_action_df['sell_price_pct'] / 100) * feature_action_df['close']  # feature_action_df.preclose
                feature_action_df['sell_volume'] = feature_action_df.apply(lambda row: row['sell_capital'] / row['sell_price'] if row['sell_price'] <= row['high'] and row['sell_price']!=0 else 0.0, axis=1)
                # make a deal
                feature_action_df['sell_volume'] = feature_action_df[['sell_volume', 'held_volume']].min(axis=1)
                feature_action_df['held_volume'] -= feature_action_df['sell_volume']
                capital += (feature_action_df['sell_volume'] * feature_action_df['sell_price']).sum()
                #print('sell',feature_action_df[['held_volume','close']])
                
                ## buy
                feature_action_df.loc[feature_action_df['buy_trading_signals']<0.5, 'buy_probability'] = 0.0
                # 新增间隔日推荐
                feature_action_df['buy_probability'] = feature_action_df['buy_probability'] * feature_action_df['rear_next_rise_pct_pred']
                
                feature_action_df['buy_probability'] = feature_action_df['buy_probability'] / feature_action_df['buy_probability'].sum()
                
                buy_positions_pct = feature_action_df['buy_positions_pct'].mean()
                #print(f'buy_positions_pct: {buy_positions_pct: .4f}')
                buy_capital_total = valuation * buy_positions_pct
                buy_capital_total = min(buy_capital_total, capital)
                feature_action_df['buy_capital'] = feature_action_df['buy_probability'] * buy_capital_total
                feature_action_df['buy_price'] = (1 + feature_action_df['rear_low_pct_pred'] * feature_action_df['buy_price_pct'] / 100) * feature_action_df['close']  # feature_action_df.preclose
                feature_action_df['buy_volume'] = feature_action_df.apply(lambda row: row['buy_capital'] / row['buy_price'] if row['buy_price'] >= row['low'] and row['buy_price']!=0 else 0.0, axis=1)
                
                feature_action_df['held_volume'] += feature_action_df['buy_volume']# make a deal
                
                
                capital -= (feature_action_df['buy_volume'] * feature_action_df['buy_price']).sum()
                #print('buy',feature_action_df[['held_volume','close']])
                total_reward = valuation - capital_init
                if dataset_type=='test':
                    transaction_order_df_1 = feature_action_df[['code', 'code_name','low','high','close','buy_price','buy_trading_signals','buy_volume','sell_price','sell_trading_signals','sell_volume','held_volume']]
                    transaction_order_df_1[['primary_key','date','capital','valuation', 'held_volume_pct','insert_timestamp','dataset_type','total_reward']] = primary_key_order, trading_day, capital, valuation, held_volume_pct, insert_timestamp, 'test', total_reward
                    #print('transaction_order_df_1',transaction_order_df_1)
                    transaction_order_list.append(transaction_order_df_1)
                    
                    #[['action','volume','state']]
                #,'total_shares_held_pct','position_pct',,'price''shares_held',
# =============================================================================
# feature_action_df.columns = Index(['code', 'tradeStatus', 'code_name', 'insert_timestamp', 'index',
#        'rear_low_pct_pred', 'rear_high_pct_pred', 'high', 'low', 'close',
#        'preclose', 'buy_price_pct', 'buy_trading_signals', 'buy_probability',
#        'buy_positions_pct', 'sell_price_pct', 'sell_trading_signals',
#        'sell_probability', 'sell_positions_pct', 'buy_capital', 'buy_price', 'buy_volume'],
#       dtype='object')
# =============================================================================
                #feature_action_df1.buy_probability
                
                #feature_action_df['minimum_transaction_amount'] = feature_action_df['buy_price'] * 100
                
                #.sort_values(by='index_start',ascending=False)
                #buy_signals_df = .sort_values(by='buy_trading_signals', ascending=False).head(10)
                
                #buy_signals_df['pre_order_amount'] = buy_signals_df['rear_low_pct_pred'] * 
                
                
                
                #rear_low_pct_pred, rear_high_pct_pred, high, low, close, preclose, date = data_raw.loc[data_raw.date==trading_day, ['rear_low_pct_pred', 'rear_high_pct_pred', 'high', 'low', 'close', 'preclose', 'date']]
                #print(data_raw.loc[data_raw.date==trading_day, ['rear_low_pct_pred', 'rear_high_pct_pred', 'high', 'low', 'close', 'preclose', 'date']])
                
                #valuation = shares_held * close + capital
                
                #print(f'{trading_day}: capital: {capital:.2f} |held_volume_pct: {held_volume_pct}| total_reward: {total_reward:.2f}')
                data.loc[trading_day, ['total_held_volume_pct']] = feature_action_df['held_volume'] * feature_action_df['close'] / valuation
                #data.loc[trading_day, ['held_volume']] = feature_action_df['held_volume']
                self.all_stock['held_volume'] = feature_action_df['held_volume'].values
                
                if dataset_type=='train':
                    next_state = data.loc[data.date==trading_day, FEATURE].values
                    self.train(state, action, total_reward, next_state, False)  # Training on each step
                    state = next_state
                    

            # 查看交易信号
            # 统计大于0.5的数据的占比
            buy_signals = action_df.buy_trading_signals
            sell_signals = action_df.sell_trading_signals
            buy_signals_gt_05_pct = round((buy_signals>=0.5).sum()/action_df.shape[0],3)
            sell_signals_gt_05_pct = round((sell_signals>=0.5).sum()/action_df.shape[0],3)
            print(f'buy_signals: min: {buy_signals.min()} |mean: {buy_signals.mean():.3f} |max: {buy_signals.max()} |>=0.5_pct: {buy_signals_gt_05_pct}')
            print(f'sell_signals: min: {sell_signals.min()} |mean: {sell_signals.mean():.3f} |max: {sell_signals.max()} |>=0.5_pct: {sell_signals_gt_05_pct}')
            if dataset_type=='train':

                
                self.save_model(primary_key_order)  # Save the model after each episode
# =============================================================================
#                 trade_share_register_list.append({'primary_key': primary_key,
#                                                   'date': date,
#                                                   'capital': capital,
#                                                   'shares_held': shares_held,
#                                                   'valuation': valuation,
#                                                   'insert_timestamp': insert_timestamp,
#                                                   }.copy())
# =============================================================================
            elif dataset_type=='test':
                ...
                #valuation = capital_init
                #total_reward = 0
                #held_volume_pct = 0
            annualized_income = trade_eval.calculate_annualized_return(beginning_value=capital_init, ending_value=valuation, number_of_days=len(date_sorted))
            
            trading_model_dict.update({'annualized_income': round(annualized_income * 100, 2),
                                       'total_reward': total_reward,
                                       'valuation': valuation,
                                       'capital': capital,
                                       })
            trading_model_list.append(trading_model_dict.copy())
        
            print(f'capital: {capital:.2f} |held_volume_pct: {held_volume_pct:.2f} |annualized_income: {annualized_income * 100:.2f}% |total_reward: {total_reward:.2f}')
            
            
            trading_model_df = pd.DataFrame(trading_model_list)
            #trade_share_register_df = pd.DataFrame(trade_share_register_list)
            with base_connect_database.engine_conn("POSTGRES") as conn:
                if dataset_type=='train':
                    trading_model_df.to_sql(TRADE_MODEL_TABLE_NAME, con=conn.engine, index=False, if_exists='append')
                if dataset_type=='test':
                    transaction_order_df = pd.concat(transaction_order_list, axis=0)
                    
                    #transaction_order_df.to_sql(TRADE_ORDER_TABLE_NAME, con=conn.engine, index=False, if_exists='append')  # 量太大了,训练阶段不输出
                
                    
                    return transaction_order_df
                #trade_share_register_df.to_sql(TRADE_SHARE_REGISTER_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')


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
    with base_connect_database.engine_conn("POSTGRES") as conn:
        sql = f"SELECT * FROM {RE_ENVIRONMENT} WHERE date >= '{date_start}' AND date < '{date_end}'"#rl_environment
        backtest_raw_df = pd.read_sql(sql, con=conn.engine)
    
    backtest_raw_df = backtest_raw_df.drop(columns=['code_name', 'insert_timestamp'])
    backtest_raw_df['total_held_volume_pct'] = 0.0
    backtest_raw_df = backtest_raw_df.drop_duplicates(['date','code'], keep='first') # 处理测试导致的重复数据污染
    #backtest_raw_df.index = backtest_raw_df['code'] + '_' + backtest_raw_df['date']
    #backtest_raw_df = backtest_raw_df.sort_values(by='date',ascending=True).reset_index(drop=True)
    
    with base_connect_database.engine_conn("POSTGRES") as conn:
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
        
    backtest_raw_df = backtest_raw_df.groupby('date').apply(__apply_completion, all_stock).reset_index(drop=True)
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
    
    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
# =============================================================================
#     # Retrieve stock ticker training round
#     with base_connect_database.engine_conn("POSTGRES") as conn:
#         all_stock = pd.read_sql("SELECT * FROM all_stock", con=conn.engine)
#         
#         all_stock['insert_timestamp'] = datetime.now().strftime('%F %T')
#         all_stock.to_sql('all_stock_copy', con=conn.engine, index=False, if_exists='replace')
# =============================================================================


    train_process(date_start=args.date_start, date_end=args.date_end)


#Position ratio 头寸比例
#['pre_sale_share']