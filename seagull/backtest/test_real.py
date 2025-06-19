# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:27:13 2024

@author: awei
"""
import pandas as pd

import efinance as ef
from __init__ import path
freq_dict = {
    5: '5min',
    15: '15min',
    30: '30min',
    60: '60min',
    101: '日线',
    102: '周线',
    103: '月线',
}
etf_freq_dict = ef.stock.get_quote_history(list(freq_dict.keys()), klt=str(15))
etf_freq_df = pd.concat({k: v for k, v in etf_freq_dict.items()}).reset_index(drop=True)
etf_freq_df.to_csv(f'{path}/_file/15_etf_freq_df.csv')