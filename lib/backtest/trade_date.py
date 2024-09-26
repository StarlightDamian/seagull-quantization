# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:24:41 2024

@author: awei
trade_date
股票交易日历	
来源：深交所
更新周期：每年初
"""
import adata

res_df = adata.stock.info.trade_calendar()	
print(res_df)
