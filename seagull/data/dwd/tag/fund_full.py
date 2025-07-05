# -*- coding: utf-8 -*-
"""
@Date: 2025/6/25 23:06
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_full.py
@Description: 
"""

with utils_database.engine_conn("POSTGRES") as conn:
    result = pd.read_sql('ods_api_baostock_stock_industry', con=conn.engine)
result = result.rename(columns={'tradeStatus': 'trade_status'})
filename = 'dwd_stock_label'

utils_data.output_database(result, filename)