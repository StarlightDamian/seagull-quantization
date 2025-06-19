# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:17:23 2024

@author: awei
"""
import pandas as pd
from __init__ import path
from base import  base_utils
history_day = pd.read_csv(f'{path}/data/history_day_board_df_1.csv')


history_day['primary_key2'] = (history_day['date']+history_day['code']).apply(base_utils.md5_str)