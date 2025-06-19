# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:40:29 2025

@author: Damian
两融数据(ods_acct_incr_akshare_margin_trading)
Margin Trading（融资交易）：投资者向券商借款买入股票。
Securities Lending（融券交易）：投资者向券商借股票卖出，以期在价格下跌时回购赚取差价。

在数据字段中常用的词：

Margin Purchase Amount（融资买入额）
Margin Purchase Balance（融资余额）
Securities Lending Amount（融券卖出量）
Securities Lending Balance（融券余额）
Margin Trading Volume（融资交易量）
Short Selling Volume（融券卖空量）

macro_china_market_margin_sh
"""

import pandas as pd

import akshare as ak
# 获取某日沪深两融数据
df = ak.stock_margin_sse(start_date="20230101", end_date="20231001")