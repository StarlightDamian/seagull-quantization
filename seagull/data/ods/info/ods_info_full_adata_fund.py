# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: base.py
@Description:
"""
import adata

from __init__ import path
from utils import utils_database, utils_log, utils_data

df = adata.fund.info.all_etf_exchange_traded_info()
print(df)

utils_data.output_database(df,
                           filename='ods_info_full_adata_fund_base')
