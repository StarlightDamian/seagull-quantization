# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:19:08 2023

@author: awei
#数据分发
"""

import pandas as pd
path = 'E:/3_software_engineering/github/quantitative-finance'

def _apply_data_distribution_day(substring_pd):
    year_month_day = substring_pd['date'].values[0]
    #year, month, day = year_month_day[:4], year_month_day[5: 7], year_month_day[8:]
    #substring_pd.reset_index().to_feather(f'{PATH}/data/day/{year}_{month}_{day}.feather')
    substring_pd.reset_index().to_feather(f'{PATH}/data/day/{year_month_day}.feather')
    
def data_distribution_day(data_day_all):
    data_day_all.groupby('date').apply(_apply_data_distribution_day)  # 按日
    
    
if __name__ == '__main__':
    data_day_all = pd.read_feather(f'{PATH}/data/k_day_19901219_20230808.feather')
    
    #数据分发
    data_distribution_day(data_day_all)
    
    