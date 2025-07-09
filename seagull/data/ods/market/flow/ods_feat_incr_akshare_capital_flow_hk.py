# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:28:18 2025

@author: Damian

资金流向-香港北向(ods_feat_incr_akshare_capital_flow_hk)

"""


import akshare as ak

stock_hsgt_fund_flow_summary_em_df = ak.stock_hsgt_fund_flow_summary_em()
print(stock_hsgt_fund_flow_summary_em_df)
