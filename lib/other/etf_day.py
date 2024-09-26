# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:01:05 2024

@author: awei
etf_day
"""
import adata
import pandas as pd
import efinance as ef
etf_code_df = adata.fund.info.all_etf_exchange_traded_info()
#etf_code_df = pd.read_sql('ods_adata_etf_code', con=self.conn.engine)
etf_dict = ef.stock.get_quote_history('etf_code_df.fund_code',beg='19900101')
result = pd.concat({k: pd.DataFrame(v) for k, v in etf_dict.items()})