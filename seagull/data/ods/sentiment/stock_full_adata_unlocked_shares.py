# -*- coding: utf-8 -*-
"""
@Date: 2024/10/30 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_full_adata_unlocked_shares.py
@Description: 股票解禁数据(ods/sentiment/stock_full_adata_unlocked_shares)
@Update cycle: day
1.最近一个月
"""

import adata

from seagull.settings import PATH
from seagull.utils import utils_data, utils_database

ods_adata_stock_lifting = adata.sentiment.stock_lifting_last_month()
utils_data.output_database(ods_adata_stock_lifting,
                           filename='ods_sentiment_stock_full_adata_unlocked_shares',
                           if_exists='append')


