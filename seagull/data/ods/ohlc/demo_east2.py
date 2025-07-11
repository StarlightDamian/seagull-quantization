# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 9:50
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_east2.py
@Description: 
"""
import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_em_5min(code: str,
                  market: int,
                  start: str,
                  end: str) -> pd.DataFrame:
    """
    拉取 [start,end] 区间的 5 分钟 K 线，自动分段拼接。

    code: 6 位代码，不带后缀
    market: 0=上交所, 1=深交所
    start,end: 'YYYY-MM-DD'
    """
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    secid = f"{market}.{code}"
    # 下面两个字段决定返回哪几列
    fields2 = "f51,f52,f53,f54,f55"  # open,close,high,low,volume

    all_records = []
    # 东财大约能一次给你 800 根左右的 5 分钟数据，折合 800*5min≈4000min≈2.7天
    # 为了保险，分段用 60 天拉一次
    dt_start = datetime.strptime(start, "%Y-%m-%d")
    dt_end = datetime.strptime(end, "%Y-%m-%d")
    delta = timedelta(days=60)

    cur_start = dt_start
    while cur_start < dt_end:
        cur_end = min(cur_start + delta, dt_end)
        params = {
            "secid": secid,
            "klt": 5,
            "fqt": 1,  # 前复权
            "beg": cur_start.strftime("%Y%m%d"),
            "end": cur_end.strftime("%Y%m%d"),
            "fields1": "f1,f2,f3,f4,f5",
            "fields2": fields2,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", {}) or {}
        klines = data.get("klines", [])
        # 解析
        for line in klines:
            dt, o, c, h, l, v = line.split(",")
            all_records.append((dt, float(o), float(c), float(h),
                                float(l), float(v)))
        # 下一段
        cur_start = cur_end + timedelta(days=1)

    # 拼成 DataFrame 并去重（跨段可能有重叠）
    df = pd.DataFrame(all_records,
                      columns=["datetime", "open", "close", "high", "low", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # 华夏上证 50ETF 510500（上交所）
    df = fetch_em_5min("510500", market=0,
                       start="2024-01-01", end="2025-07-07")
    print(df.shape)
    print(df.head(), df.tail())
