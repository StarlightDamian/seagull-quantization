# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 7:36
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_incr_akshare.py
@Description:
 "fund_etf_fund_daily_em",  # 场内交易基金-实时数据
 "fund_etf_fund_info_em",  # 场内交易基金-历史数据,['净值日期', '单位净值', '累计净值', '日增长率', '申购状态', '赎回状态']
 "fund_etf_category_sina"  # 基金实时行情-新浪
 "fund_etf_hist_sina"  # 基金行情-新浪
 "fund_etf_dividend_sina"  # 新浪财经-基金-ETF 基金-累计分红
 "fund_etf_hist_em"  # 基金历史行情-东财
 "fund_etf_hist_min_em"  # 基金分时行情-东财
 "fund_etf_spot_em"  # 基金实时行情-东财
 "fund_etf_spot_ths"  # 基金实时行情-同花顺

"""
import akshare as ak

etf_incr_df = ak.fund_etf_fund_info_em(fund="516010",
                                       start_date="20000101",
                                       end_date="20500101")
print(etf_incr_df)
print(etf_incr_df.columns)
# 筛选特定 ETF（如 516010、159869）
#target_etfs = etf_incr_df[etf_incr_df["代码"].isin(["516010", "159869"])]
#print(target_etfs[["代码", "名称", "最新价", "涨跌幅", "成交额"]])
