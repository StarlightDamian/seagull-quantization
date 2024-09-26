# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:53:13 2024

@author: awei
(data_day_1_min)
分钟级
有ETF
etf_columns=['股票名称', '股票代码', '时间', '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入']
stock_columns=['股票名称', '股票代码', '日期', '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入',
       '主力净流入占比', '小单流入净占比', '中单流入净占比', '大单流入净占比', '超大单流入净占比', '收盘价', '涨跌幅']
"""

import efinance as ef
df = ef.stock.get_today_bill('159660')
df1 = ef.stock.get_history_bill('300750')