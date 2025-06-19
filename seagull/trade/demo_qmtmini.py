# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:05:05 2024

@author: awei
demo_qmtmini
获取涨跌停价格
"""

from xtquant import xtdata
df = xtdata.get_instrument_detail('30100.sh')
print(df)