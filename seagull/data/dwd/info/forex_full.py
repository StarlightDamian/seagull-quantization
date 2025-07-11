# -*- coding: utf-8 -*-
"""
@Date: 2025/7/11 9:55
@Author: Damian
@Email: zengyuwei1995@163.com
@File: forex_full.py
@Description: 外汇
"""
import requests
import pandas as pd

API_KEY = "你的 AlphaVantage Key"

def fetch_fx_5min_av(from_sym: str, to_sym: str, outputsize="full"):
    """
    from_sym/to_sym: 例如 'EUR','USD'
    outputsize: 'compact' (~1个月)，'full' (~1年)
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "interval": "5min",
        "outputsize": outputsize,
        "apikey": API_KEY
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json().get("Time Series FX (5min)", {})
    # 转成 DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low":  "low",
        "4. close":"close"
    }).reset_index().rename(columns={"index":"datetime"})
    return df

if __name__ == "__main__":
    df = fetch_fx_5min_av("EUR","USD",outputsize="full")
    print(df.head(), df.shape)

    # # Linux/macOS
    # for i in {1..5}; do
    #   python your_script.py
    #   sleep 12  # 每 12 秒一次，确保 <5 次/分钟
    # done
