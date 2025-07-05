# -*- coding: utf-8 -*-
"""
@Date: 2024/8/27 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: bond_realtime_adata.py
@Description: 获取新浪的最新可转债行情(ods_real_bond_adata)
adata.bond.market.list_market_current()	# 获取A股市场的可转换债券最新行情
url : http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeDataSimple
"""

import adata
adata.bond.market.list_market_current()


class odsRealAdataConvertiblesBaseReptile(BondCode):
    
    ...