# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 7:57
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_incr_efinance_minute.py
@Description:
     股票名称    股票代码                日期     开盘  ...    振幅   涨跌幅    涨跌额   换手率
0     游戏ETF  159869  2025-05-23 09:35  1.109  ...  0.72 -0.36 -0.004  0.69
1     游戏ETF  159869  2025-05-23 09:40  1.109  ...  0.90 -0.54 -0.006  0.74
2     游戏ETF  159869  2025-05-23 09:45  1.103  ...  0.54  0.09  0.001  0.38
3     游戏ETF  159869  2025-05-23 09:50  1.105  ...  0.36 -0.27 -0.003  0.21
4     游戏ETF  159869  2025-05-23 09:55  1.102  ...  0.36 -0.27 -0.003  0.15
...     ...     ...               ...    ...  ...   ...   ...    ...   ...
1483  游戏ETF  159869  2025-07-07 14:40  1.278  ...  0.16 -0.08 -0.001  0.06
1484  游戏ETF  159869  2025-07-07 14:45  1.277  ...  0.16  0.16  0.002  0.10
1485  游戏ETF  159869  2025-07-07 14:50  1.279  ...  0.16  0.00  0.000  0.12
1486  游戏ETF  159869  2025-07-07 14:55  1.279  ...  0.08  0.00  0.000  0.15
1487  游戏ETF  159869  2025-07-07 15:00  1.279  ...  0.16  0.08  0.001  0.13

[1488 rows x 13 columns]

"""
import pandas as pd
import efinance as ef
from seagull.utils import utils_time, utils_data, utils_thread, utils_database, utils_character, utils_log

# ETF 代码（以中概互联网 ETF 为例）
with utils_database.engine_conn("POSTGRES") as conn:
    asset_code_df = pd.read_sql(
        "SELECT 基金代码 as asset_code FROM ods_info_fund_full_efinance where 基金类型 = 'etf'",
        con=conn.engine)
    fund_df = pd.read_sql("select distinct 股票代码 as asset_code from ods_ohlc_fund_incr_efinance_minute", con=conn.engine)

asset_code_list = asset_code_df[~(asset_code_df['asset_code'].isin(fund_df.asset_code))]['asset_code'].tolist()
print(len(asset_code_list))
fund_df_dict = ef.stock.get_quote_history(asset_code_list, klt=5, fqt=1, beg="19000101", end="20500101")
fund_df = pd.concat({asset_code: df for asset_code, df in fund_df_dict.items() if not df.empty}, ignore_index=True)
fund_df['freq_code'] = 5  # 5分钟
utils_data.output_database_large(fund_df,
                                 filename='ods_ohlc_fund_incr_efinance_minute',
                                 if_exists='append'
                                 )
