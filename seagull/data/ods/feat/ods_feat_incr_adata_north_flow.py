# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:47:19 2024

@author: awei
北向资金(ods_feat_incr_adata_north_flow_api)
"""

import adata
df = adata.sentiment.north.north_flow(start_date='1990-01-01')
print(df)
