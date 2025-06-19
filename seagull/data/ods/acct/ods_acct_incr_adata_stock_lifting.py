# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:44:49 2024

@author: awei
股票解禁数据(ods_acct_incr_adata_stock_lifting)

1.最近一个月
"""
import adata

from __init__ import path
from utils import utils_data, utils_database

ods_adata_stock_lifting = adata.sentiment.stock_lifting_last_month()
utils_data.output_database(ods_adata_stock_lifting,
                           filename='ods_acct_incr_adata_stock_lifting',
                           if_exists='append')


