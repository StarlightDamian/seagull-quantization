# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:54:16 2024

@author: awei
板块测试(test_board)
"""
import argparse

import pandas as pd

from __init__ import path
from base import base_connect_database
from test_ import test_0_lightgbm, test_2_price_limit, test_2_stock_pick, test_3_short_term_recommend


class testBoard(test_0_lightgbm.lightgbmTest):
    def __init__(self):
        self.test_price_limit = test_2_price_limit.testPriceLimit()
        self.test_stock_pick = test_2_stock_pick.testStockPick()
        self.test_short_term_recommend = test_3_short_term_recommend.testShortTermRecommend()
        
    def board_pipline(self, history_day_df):
        #global price_limit_df1,stock_recommended_df1, stock_pick_df1,short_term_recommend_df1
        
        # 一字涨跌停
        price_limit_df = self.test_price_limit.test_board_pipline(history_day_df)
        price_limit_df = price_limit_df[['primary_key', 'rear_price_limit_pred']]
        
        # 日内上升
        #stock_recommended_df = self.test_stock_recommend.test_board_pipline(history_day_df)
        #stock_recommended_df1 =stock_recommended_df
        #stock_recommended_df = stock_recommended_df[['primary_key', 'rear_rise_pct_pred']]

        # 选股
        stock_pick_df = self.test_stock_pick.test_board_pipline(history_day_df)
        #stock_pick_df1 = stock_pick_df
        
        # 间隔日上升
        history_day_df = pd.merge(history_day_df, stock_pick_df[['primary_key', 'rear_low_pct_pred', 'rear_high_pct_pred', 'rear_diff_pct_pred', 'rear_open_pct_pred', 'rear_close_pct_pred']])
        
        
        short_term_recommend_df = self.test_short_term_recommend.test_board_pipline(history_day_df)
        #short_term_recommend_df1 = short_term_recommend_df
        short_term_recommend_df = short_term_recommend_df[['primary_key', 'rear_next_rise_pct_pred','rear_next_fall_pct_pred','rear_next_pct_pred']]

        
        #stock_pick_df = stock_pick_df.drop_duplicates('primary_key',keep='first')
        #stock_recommended_df = stock_recommended_df.drop_duplicates('primary_key',keep='first')
        #price_limit_df = price_limit_df.drop_duplicates('primary_key',keep='first')
        
        
        stock_pick_df = pd.merge(stock_pick_df, price_limit_df, on='primary_key')
        #stock_pick_df = pd.merge(stock_pick_df, stock_recommended_df, on='primary_key')
        stock_pick_df = pd.merge(stock_pick_df, short_term_recommend_df, on='primary_key')
        return stock_pick_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for backtesting')
    #parser.add_argument('--date_start', type=str, default='2023-01-01', help='Start time for testing')
    parser.add_argument('--date_start', type=str, default='2024-04-10', help='Start time for testing')
    #parser.add_argument('--date_end', type=str, default='2022-01-01', help='End time for backtesting')
    parser.add_argument('--date_end', type=str, default='2024-04-15', help='End time for testing')
    args = parser.parse_args()

    print(f'Start time for testing: {args.date_start}\nEnd time for testing: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(history_day_df)
    test_board = testBoard()
    history_day_df = test_board.board_data(history_day_df)
    prediction_df = test_board.board_pipline(history_day_df)
    prediction_df.to_sql('rl_environment', con=conn.engine, index=False, if_exists='append') #replace
    
    #prediction_df.groupby('date').agg(maxid=('rear_rise_pct_pred','max'))
