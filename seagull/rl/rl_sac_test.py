# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:23:34 2023

@author: awei
SAC算法_模型结构_预测过程(rl_sac_test)

问题:
    1.self.all_stock['held_volume'] = 0.0
"""
import argparse
from datetime import datetime
import math

import pandas as pd


from seagull.settings import PATH
from trade import trade_model_eval
from reinforcement_learning import rl_sac_train
from base import base_connect_database, base_utils


TRADE_ORDER_TABLE_NAME = 'trade_order_details'
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
RE_ENVIRONMENT = 'rl_environment'  # rl_environment
class SACAgentFredict(rl_sac_train.SACAgentTrain):
    def __init__(self, state_dim, action_dim, all_stock):
        super().__init__(state_dim, action_dim)
        self.all_stock = all_stock
        

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
        sql = f"SELECT * FROM {RE_ENVIRONMENT} WHERE date >= '{date_start}' AND date < '{date_end}'"
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
        all_stock = pd.read_sql("SELECT * FROM all_stock_copy", con=conn.engine)#all_stock_copy
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
    agent = rl_sac_train.SACAgentTrain(state_dim, action_dim, all_stock)
    primary_key = trade_model_eval.annualized_income_max_primary_key()
    agent.load_model(primary_key)  # Load the model for the next episode
    trading_model_df = agent.simulate_trading(backtest_df, date_start, date_end, dataset_type='test')
    return trading_model_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2022-01-01', help='Start time for backtesting')
    parser.add_argument('--date_start', type=str, default='2022-11-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    
    #parser.add_argument('--date_start', type=str, default='2024-02-01', help='Start time for backtesting')
    #parser.add_argument('--date_end', type=str, default='2024-02-08', help='End time for backtesting')
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
    
# =============================================================================
# def train_process(date_start='2020-01-01', date_end='2022-01-01'):
#     with base_connect_database.engine_conn("POSTGRES") as conn:
#         sql = f"SELECT * FROM prediction_stock_price_test WHERE date >= '{date_start}' AND date < '{date_end}'"#2020-01-02, 2024-01-02
#         backtest_df = pd.read_sql(sql, con=conn.engine)
#     
#     #print(backtest_df)
#     #backtest_df = backtest_df.sort_values(by='date',ascending=True).reset_index(drop=True)
#     #backtest_df['total_shares_held_pct'] = 0.0
#     #backtest_df = backtest_df.drop_duplicates('date',keep='first') # 处理测试导致的重复数据污染
#     #if backtest_df.shape[0]==1:
#     #    backtest_df = pd.concat([backtest_df, backtest_df], ignore_index=True)
#     backtest_input_df = backtest_df[['rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'preclose', 'open', 'high',
#                                      'low', 'close', 'volume', 'amount', 'turn', 'pctChg', 'total_shares_held_pct']]
#     
#     # Initialize agent and environment
#     state_dim = len(FEATURE)
#     action_dim = 8
#     agent = SACAgentFredict(state_dim, action_dim)
#     agent.simulate_trading(backtest_input_df, date_start, date_end)
#     
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--date_start', type=str, default='2022-01-01', help='Start time for backtesting')
#     parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
#     #parser.add_argument('--date_start', type=str, default='2023-06-01', help='Start time for backtesting')
#     #parser.add_argument('--date_end', type=str, default='2023-12-23', help='End time for backtesting')
#     args = parser.parse_args()
# =============================================================================
    
# =============================================================================
#     #rl_sac_train.train_process(code='sz.002230', date_start=args.date_start, date_end=args.date_end)
#     stock_dict = {'sz.002281': '光迅科技',
#                   'sz.002747': '埃斯顿',
#                   'sh.600522': '中天科技',
#                   'sh.600732': '爱旭股份',
#                   }
# 
#     for code in stock_dict.keys():
#         train_process(code=code, date_start=args.date_start, date_end=args.date_end)
# =============================================================================
    
    
# =============================================================================
#     def simulate_trading(self, data, code, date_start, date_end):
#         #trading_model_list = []#, []trade_share_register_list
#         transaction_order_list = []
#         #episodes = 3
#         #for episode in range(episodes):  # Simulate trading for 100 episodes
#         #print(f'{episode} / {episodes}')
#         
#         # Initial
#         capital_init = 1_000_000
#         capital = capital_init
#         total_shares_held_pct = 0
#         #  shares_held = 0 # debug 没有卖出价格
#         shares_held = 10_000
#         state = data.iloc[0].values
#         insert_timestamp = datetime.now().strftime('%F %T')
#         #primary_key = base_utils.md5_str(insert_timestamp)
#         
# # =============================================================================
# #         trading_model_dict = {'primary_key': primary_key,
# #                               #'episode': episode,
# #                               'capital_init': capital_init,
# #                               'insert_timestamp': insert_timestamp,
# #                               }
# # =============================================================================
#         
# 
#         
#         for trading_day in range(1, len(data)):
#                 #print(f'{trading_day}: capital: {capital:.2f}')
#                 # sell_ratio % in valume
#                 # buy_ratio % in capital
#                 
#             action = self.get_action(state)
#             #print('action',action)
#             buy_ratio, buy_price_ratio, buy_trading_signals, sell_ratio, sell_price_ratio, sell_trading_signals = action  # action_dim = 4
#             
#             # Because sigmoid reduces the confidence interval to [0,1] and enlarges it to [0,2]
#             buy_price_ratio = buy_price_ratio * 2
#             sell_price_ratio =  sell_price_ratio * 2
#             #print(buy_ratio, buy_price_ratio, sell_ratio, sell_price_ratio)
#             #preclose = data['close'].iloc[trading_day - 1]
#             rear_low_pct_pred, rear_high_pct_pred, high, low, close, preclose, date = data_raw.loc[trading_day, ['rear_low_pct_pred', 'rear_high_pct_pred', 'high', 'low', 'close', 'preclose', 'date']]
#             #print()
#             valuation = shares_held * close + capital
#             total_reward = valuation - capital_init
#             total_shares_held_pct = shares_held * close / valuation
#             data.loc[trading_day, ['total_shares_held_pct']] = total_shares_held_pct
#             transaction_order_dict = {'primary_key': primary_key,
#                                       'code': code,
#                                       'date': date,
#                                       'capital': capital,
#                                       'shares_held': shares_held,
#                                       'valuation': valuation,
#                                       'total_shares_held_pct': total_shares_held_pct,
#                                       'total_reward': total_reward,
#                                       'low': low,
#                                       'high': high,
#                                       'close': close,
#                                       'dataset_type': 'test',
#                                       'insert_timestamp': insert_timestamp,
#                                       }
#             
#             # Buy/Sell decision based on SAC agent's action
#             # pending order
#             if sell_trading_signals > 0.5:
#                 sell_price = (1 + rear_high_pct_pred * sell_price_ratio / 100) * preclose
#                 if (sell_ratio > 0) and (shares_held > 0): # The selling price is less than or equal to the highest price
#                     transaction_order_dict.update({'action': 'sell', 'price': round(sell_price, 2)})
#                     # trade
#                     if sell_price <= high:
#                         volume = shares_held if shares_held<100 else math.floor(sell_ratio * shares_held)  # Less than 100 shares can only be sold in full
#                         capital += (volume * sell_price)
#                         position_pct = volume / shares_held
#                         shares_held -= volume
#                         # print(f'sell_date: {date} |shares_held:{shares_held} |volume: {volume} |sell_price: {sell_price:.2f} |high: {high}')
#                         transaction_order_dict.update({'volume': volume, 'shares_held': shares_held, 'state': 'make_a_deal', 'position_pct': round(position_pct * 100, 2)})
#                         
#                         
#                     elif sell_price > high:
#                         transaction_order_dict.update({'volume': 0, 'state': 'withdraw', 'position_pct': 0})
#                     transaction_order_list.append(transaction_order_dict.copy())  # The dict added to the list can still be modified, so add copy
#                     
#             
#             # pending order
#             if buy_trading_signals > 0.5:
#                 buy_price = (1 + rear_low_pct_pred * buy_price_ratio / 100) * preclose
#                 if (buy_ratio > 0) and (buy_ratio * capital >= buy_price * 100):  # Need to buy at least 100 shares, 
#                     transaction_order_dict.update({'action': 'buy', 'price': round(buy_price, 2)})
#                     # trade
#                     if buy_price >= low:
#                         volume = math.floor((buy_ratio * capital) / buy_price)
#                         shares_held += volume
#                         position_pct = volume * buy_price / capital
#                         capital -= (volume * buy_price)
#                         #print(f'buy_date: {date} |shares_held:{shares_held} |volume: {volume} |buy_price: {buy_price:.2f} |low: {low}')
#                         transaction_order_dict.update({'volume': volume, 'shares_held': shares_held, 'state': 'make_a_deal', 'position_pct': round(position_pct * 100, 2)})
#                         
#                     elif buy_price < low:
#                         transaction_order_dict.update({'volume': 0, 'state': 'withdraw', 'position_pct': 0})
#                     transaction_order_list.append(transaction_order_dict.copy())
#         
#         if data.shape[0]!=1:  # 用于预测真实数据
#             next_state = data.iloc[trading_day].values
#             self.train(state, action, total_reward, next_state, False)  # Training on each step
#             state = next_state
#             
#             self.save_model(primary_key)  # Save the model after each episode
# # =============================================================================
# #                 trade_share_register_list.append({'primary_key': primary_key,
# #                                                   'date': date,
# #                                                   'capital': capital,
# #                                                   'shares_held': shares_held,
# #                                                   'valuation': valuation,
# #                                                   'insert_timestamp': insert_timestamp,
# #                                                   }.copy())
# # =============================================================================
#             
#         #annualized_income = trade_eval.calculate_annualized_return(beginning_value=capital_init, ending_value=valuation, number_of_days=len(data))
#         
# # =============================================================================
# #         trading_model_dict.update({'total_reward': total_reward,
# #                                    'valuation': valuation,
# #                                    'capital': capital,
# #                                    'shares_held': shares_held,
# #                                    'annualized_income': round(annualized_income * 100, 2),
# #                                    })
# # =============================================================================
#         #trading_model_list.append(trading_model_dict.copy())
#         
#         
#         transaction_order_df = pd.DataFrame(transaction_order_list)
#         #trading_model_df = pd.DataFrame(trading_model_list)
#         #trade_share_register_df = pd.DataFrame(trade_share_register_list)
#         with base_connect_database.engine_conn("POSTGRES") as conn:
#             print('transaction_order_df',transaction_order_df)
#             transaction_order_df.to_sql(TRADE_ORDER_TABLE_NAME, con=conn.engine, index=False, if_exists='append')  # append
#             #trading_model_df.to_sql(TRADE_MODEL_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')
#             #trade_share_register_df.to_sql(TRADE_SHARE_REGISTER_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')
#     
# =============================================================================
