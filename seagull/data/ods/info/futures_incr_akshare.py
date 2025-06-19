# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: futures_incr_akshare.py
@Description: 期货(ods/info/futures_incr_akshare)

df.columns = ['交易所', '品种', '代码', '交易保证金比例', '涨跌停板幅度', '合约乘数', '最小变动价位', '限价单每笔最大下单手数',
       '特殊合约参数调整', '调整备注']
"""

import akshare as ak
futures_rule_df = ak.futures_rule(date="20250513")
print(futures_rule_df)
