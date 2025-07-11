# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 9:43
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_east.py
@Description:
1492

2025-05-23
"""
import requests
import pandas as pd
from datetime import datetime

def fetch_etf_5min(code: str, market: int = 0, count: int = 100):
    """
    code: ETF 6 位代码，如 '510300'
    market: 0=上证，1=深证
    count: 最近多少根 K 线
    """
    secid = f"{market}.{code}"
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "klt": 5,
        "fqt": 1,
        "beg": 20250501,
        "end": 20250601,
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55",
    }
    resp = requests.get(url, params=params, timeout=5)
    data = resp.json().get("data", {})
    klines = data.get("klines", [])[-count:]
    # 每条 klines 格式 "YYYY-MM-DD HH:MM:SS,open,close,high,low,volume"
    records = [line.split(",") for line in klines]
    print(records)
    df = pd.DataFrame(records, columns=["datetime","open","close","high","low"])#,"volume"
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open","close","high","low"]:#,"volume"
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

if __name__ == "__main__":
    # 例：华夏上证 50 ETF（510500 在上交所）
    df = fetch_etf_5min("159869", market=0, count=10000)
    print(df)
