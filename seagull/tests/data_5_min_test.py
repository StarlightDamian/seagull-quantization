# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 03:26:28 2024

@author: awei
"""

#5分钟线
import baostock as bs
bs.login()
k_rs = bs.query_history_k_data_plus('sh.600000',
                                    "date,code,open,high,low,close",
                                    "2022-01-01",
                                    "2022-10-10",
                                    frequency='5')
bs.logout()
print(k_rs.get_data())
