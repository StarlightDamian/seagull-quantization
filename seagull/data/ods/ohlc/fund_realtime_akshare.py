# -*- coding: utf-8 -*-
"""
@Date       : 2025/6/18 12:54
@Author     : Damian
@Email      : zengyuwei1995@163.com
@File       : fund_realtime_akshare.py
@Description: (ods_real_fund_akshare)
"""
import akshare as ak

# 获取全市场 ETF 实时行情
etf_realtime_df = ak.fund_etf_spot_em()

# 筛选特定 ETF（如 516010、159869）
target_etfs = etf_realtime_df[etf_realtime_df["代码"].isin(["516010", "159869"])]
print(target_etfs[["代码", "名称", "最新价", "涨跌幅", "成交额"]])
