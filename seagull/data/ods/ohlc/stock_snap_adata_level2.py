# -*- coding: utf-8 -*-
"""
@Date: 2024/10/23 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_snap_adata_level2.py
@Description: 五档行情(ods/ohlc/stock_snap_adata_level2)
https://finance.pae.baidu.com/vapi/v1/getquotation?srcid=5353&all=1&pointType=string&group=quotation_minute_ab&query=872925&code=872925&market_type=ab&newFormat=1&name=锦好医疗&finClientType=pc
"""
import adata
# ['000001', '600001', '000795', '872925']#'000001'
df = adata.stock.market.get_market_five()
print(df)

