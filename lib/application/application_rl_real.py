# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:09:45 2023

@author: awei
真实数据预测(application_rl_real)
俞老师量化_2024-03-01
数据来源日期:2024-03-01    数据预测日期: 2024-03-04
"""
import argparse

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from __init__ import path
from reinforcement_learning import rl_sac_test
from base import base_connect_database, base_arguments, base_trading_day

pd.set_option('display.max_columns', 15)  # 显示 10 列
scaler_0_10 = MinMaxScaler(feature_range=(0, 10))
scaler_0_1 = MinMaxScaler(feature_range=(0, 1))
RE_ENVIRONMENT = 'rl_environment'

def excel():
    writer = pd.ExcelWriter(f'{path}/wechat/output_stock_forecast.xlsx')
    output_prediction_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    
def prediction_target_date(target_date):
    with base_connect_database.engine_conn('postgre') as conn:
        prediction_merge_df = pd.read_sql(f"SELECT * FROM {RE_ENVIRONMENT} WHERE date = '{target_date}'", con=conn.engine)
        stock_industry = pd.read_sql('stock_industry', con=conn.engine)
    prediction_merge_df = prediction_merge_df.drop_duplicates(['primary_key'], keep='first')
    
    prediction_merge_df = pd.merge(prediction_merge_df,stock_industry[['code', 'industry']], on='code')
    
    prediction_merge_df['rear_open_pred'] =round(prediction_merge_df['close'] * (1+prediction_merge_df['rear_open_pct_pred'] / 100), 3)
    
    prediction_merge_df['rear_low_pred'] =round(prediction_merge_df['close'] * (1+prediction_merge_df['rear_low_pct_pred'] / 100), 3)
    
    prediction_merge_df['rear_high_pred'] =round(prediction_merge_df['close'] * (1+prediction_merge_df['rear_high_pct_pred'] / 100), 3)

    prediction_merge_df['rear_close_pred'] =round(prediction_merge_df['close'] * (1+prediction_merge_df['rear_close_pct_pred'] / 100), 3)
    
    #prediction_merge_df['rear_close_pct_pred'] = scaler_0_10.fit_transform(prediction_merge_df['rear_close_pct_pred'].values.reshape(-1, 1))
    #prediction_merge_df['rear_rise_pct_pred'] = scaler_0_10.fit_transform(prediction_merge_df['rear_rise_pct_pred'].values.reshape(-1, 1))
    #prediction_merge_df['vote_index'] = prediction_merge_df['rear_rise_pct_pred'] + prediction_merge_df['rear_close_pct_pred']
    #prediction_merge_df['vote_index'] = prediction_merge_df['rear_next_rise_pct_pred'] -prediction_merge_df['rear_next_fall_pct_pred']
    prediction_merge_df['vote_index'] = prediction_merge_df['rear_next_pct_pred']
    
    prediction_merge_df['remark'] = prediction_merge_df.rear_price_limit_pred.map({1: '一字涨跌停,',
                                                                  0: ''})
    prediction_merge_df = prediction_merge_df.sort_values(by='vote_index', ascending=False)
    return prediction_merge_df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_date', type=str, default='2024-04-12', help='Start time for training')
    args = parser.parse_args()
    
    prediction_merge_raw_df = prediction_target_date(args.target_date)
    prediction_merge_df = prediction_merge_raw_df[['code', 'industry', 'rear_open_pred', 'rear_low_pred', 'rear_high_pred', 'rear_close_pred', 'vote_index', 'remark']]#,'rear_diff_pct_pred', 'rear_close_pct_pred', 'rear_rise_pct_pred','rear_price_limit_pred'
                                              
    transaction_order_df = rl_sac_test.train_process(date_start=args.target_date, date_end='2099-12-31')
    #global transaction_order_df1
    #transaction_order_df1 = transaction_order_df
    
    
    transaction_order_raw_df = transaction_order_df.loc[transaction_order_df.date==args.target_date]
    transaction_order_df = transaction_order_raw_df[['code','code_name','low', 'high', 'close', 'buy_price', 'buy_trading_signals', 'sell_price', 'sell_trading_signals']]
    transaction_order_df['buy_signals'] = scaler_0_1.fit_transform(transaction_order_df['buy_trading_signals'].values.reshape(-1, 1))
    transaction_order_df['sell_signals'] = scaler_0_1.fit_transform(transaction_order_df['sell_trading_signals'].values.reshape(-1, 1))
    transaction_order_df['buy_signals'] = transaction_order_df['buy_signals'].apply(lambda x: 1 if x > 0.5 else 0)
    transaction_order_df['sell_signals'] = transaction_order_df['sell_signals'].apply(lambda x: 1 if x > 0.5 else 0)
    
    #transaction_order_df[['sell_price', 'buy_price']] = transaction_order_df[['sell_price', 'buy_price']].round(3)
    
    prediction_df = pd.merge(prediction_merge_df, transaction_order_df, on='code')
    
    output_prediction_df = prediction_df[['code','code_name','industry','low', 'high', 'close', 'rear_open_pred', 'rear_low_pred', 'rear_high_pred', 'rear_close_pred', 'vote_index', 'remark']]#, 'buy_price',  'sell_price'
    
    output_prediction_df = output_prediction_df.rename(
        columns={'code': '股票代码',
                 'code_name': '公司名称',
                 'industry': '行业',
                 'low': '最低价',
                 'high': '最高价',
                 'close': '收盘价',
                 'rear_open_pred': '开盘价',
                 'rear_low_pred': '最低价',
                 'rear_high_pred': '最高价',
                 #'buy_price': '建议买入价格',
                 #'sell_price': '建议卖出价格',                                                              
                 'rear_close_pred': '收盘价',
                 'vote_index': '推荐指数',
                 'remark': '备注',
                 })
    
    target_date_replace = args.target_date.replace('-', '')
    output_prediction_df.to_csv(f'{path}/wechat/俞老师量化_{target_date_replace}.csv', index=False)
    
    trading_day = base_trading_day.tradingDay()
    trading_day_pre_dict = trading_day.specified_trading_day_after()
    trading_day_next = trading_day_pre_dict[args.target_date]
    print(f'俞老师量化_{target_date_replace}')
    print(f'数据来源日期:{args.target_date}    数据预测日期: {trading_day_next}')
    
    for user, code_name_list in base_arguments.argparse_user_dict.items():
        output_personal = output_prediction_df.loc[output_prediction_df['公司名称'].isin(code_name_list)]
        output_personal.to_csv(f'{path}/wechat/俞老师量化_{user}_{target_date_replace}.csv')
        
        
def trade_order_details():
    ...
     #trade_order_details_df.columns = ['primary_key', 'code', 'date', 'capital', 'shares_held', 'valuation','total_shares_held_pct', 'total_reward', 'low', 'high', 'close','dataset_type', 'insert_timestamp', 'action', 'price', 'volume','state', 'position_pct']

    #transaction_order_df = transaction_order_df[['primary_key', 'primary_key_order', 'dataset_type', 'episode', 'annualized_income', 'valuation', 'capital', 'total_reward', 'capital_init', 'lr_actor', 'lr_critic', 'date_start', 'date_end', 'insert_timestamp']]