# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 01:50:22 2024

@author: awei
分板块训练(train_board)
"""
import argparse

import pandas as pd

from seagull.settings import PATH
from train import train_0_lightgbm, train_2_price_limit, train_2_stock_pick, train_3_short_term_recommend
from base import base_connect_database

class trainBoard(train_0_lightgbm.lightgbmTrain):
    def __init__(self):
        self.train_price_limit = train_2_price_limit.trainPriceLimit()
        self.train_stock_pick = train_2_stock_pick.trainStockPick()
        self.train_stock_term_recommend = train_3_short_term_recommend.trainShortTermRecommend()
        
    def board_2_pipline(self, history_day_df):
        self.train_price_limit.train_board_pipline(history_day_df)
        self.train_stock_pick.train_board_pipline(history_day_df)
        
    def board_3_pipline(self, history_day_df):
        self.train_stock_term_recommend.train_board_pipline(history_day_df)
        
if __name__ == '__main__':
    
    train_board = trainBoard()
# =============================================================================
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--date_start', type=str, default='2014-01-01', help='Start time for training')
#     #parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for training')
#     parser.add_argument('--date_end', type=str, default='2020-01-01', help='end time of training')
#     args = parser.parse_args()
# 
#     print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
#     
#     # Load date range data
#     with base_connect_database.engine_conn("POSTGRES") as conn:
#         history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
#     
#     
#     history_day_df = train_board.board_data(history_day_df)
#     train_board.board_2_pipline(history_day_df)
# =============================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for training')
    #parser.add_argument('--date_start', type=str, default='2019-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    history_day_short_term_df = train_3_short_term_recommend.feature_engineering_short_term_recommend(args.date_start, args.date_end)
    history_day_short_term_df = train_board.board_data(history_day_short_term_df)
    train_board.board_3_pipline(history_day_short_term_df)