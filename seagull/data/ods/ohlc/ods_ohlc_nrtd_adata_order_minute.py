# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:46:01 2024

@author: awei
高频分析
获取单个股票的成交分时，最新200条记录
(ods_ohlc_nrtd_adata_order_minute)
"""
import adata
df = adata.stock.market.get_market_bar(stock_code='000001')
print(df)



# =============================================================================
# stock_code	string	代码	600001
# trade_time	datetime	成交时间	2023-09-13 09:31:45
# price	decimal	当前价格(元)	12.36
# volume	decimal	成交量(股)	34452500
# bs_type	string	买卖类型	B：买入，S：卖出
# =============================================================================
