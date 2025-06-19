# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:41:18 2025

@author: awei
demo_feather
"""
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database

if __name__ == '__main__':
# =============================================================================
#     data = {'A':1,'B':2}
#     df = pd.DataFrame([data])
#     print(df)
# =============================================================================
    with utils_database.engine_conn("POSTGRES") as conn:
        raw_df = pd.read_sql("das_wide_incr_train", con=conn.engine)
    
    raw_df = raw_df.loc[~(raw_df.date==raw_df.date.max())]
    
    
    output_df = raw_df.loc[raw_df.date>='2024-10-01',['date','full_code','board_primary_key','open','high','low','close','volume','close_rate','turnover','vwap']]
    output_df.reset_index(drop=True).to_feather(f'{PATH}/data/das_wide_incr_train_mini.feather')

    #raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train.feather')
