# -*- coding: utf-8 -*-
"""
@Date: 2024/10/29 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_snap_adata_order_minute.py
@Description: 高频分析(ods/ohlc/stock_snap_adata_order_minute)
@Update cycle: minute
获取单个股票的成交分时，最新200条记录
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
