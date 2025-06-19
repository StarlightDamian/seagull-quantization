# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_full_adata.py
@Description: (ods/info/fund_full_adata)
@Update cycle: day
"""
import adata

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_data

df = adata.fund.info.all_etf_exchange_traded_info()
print(df)

utils_data.output_database(df,
                           filename='ods_info_full_adata_fund_base')

df = adata.fund.info.all_etf_exchange_traded_info()
print(df)
