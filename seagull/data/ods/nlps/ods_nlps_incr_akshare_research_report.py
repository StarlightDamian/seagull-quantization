# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:48:35 2025

@author: Damian
研报(ods_nlps_incr_akshare_research_report)
"""

import akshare as ak

stock_research_report_em_df = ak.stock_research_report_em(symbol="000001")
print(stock_research_report_em_df)