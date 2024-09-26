# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 00:50:55 2024

@author: awei
data_ods_info_real_adata_bond_base
adata.bond.market.list_market_current()	# 获取A股市场的可转换债券最新行情
获取新浪的最新可转债行情
url : http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeDataSimple
"""
import adata
adata.bond.market.list_market_current()


class odsRealAdataConvertiblesBaseReptile(BondCode):
    
    