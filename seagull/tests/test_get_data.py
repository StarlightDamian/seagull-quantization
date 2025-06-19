# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:20:33 2023

@author: awei
"""
#7点可以

import baostock as bs
bs.login()
k_rs = bs.query_history_k_data_plus('sh.512170', "date,code,open,high,low,close", "2023-12-29", "2023-12-29")
bs.logout()
print(k_rs.get_data())
