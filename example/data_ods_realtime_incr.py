# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:46:07 2024

@author: awei
(data_api_realtime)
"""
from seagull.settings import PATH
import data_api_efinance
import pprint

def custom_activation(x):
    if x > 0:
        return x
    else:
        return -2 * x

if __name__ == '__main__':
    stock_code = ['000560','603062','512760','512480']
    realtime_df = data_api_efinance.realtime(stock_code)
    print(realtime_df.values)
    
    last_df = data_api_efinance.last(stock_code)
    print(last_df.values)
    
    
    