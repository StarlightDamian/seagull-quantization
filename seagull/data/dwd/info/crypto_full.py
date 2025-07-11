# -*- coding: utf-8 -*-
"""
@Date: 2025/7/11 9:52
@Author: Damian
@Email: zengyuwei1995@163.com
@File: crypto_full.py
@Description: 加密货币
"""
import ccxt
import pandas as pd
from datetime import datetime

# 1) 初始化交易所（以 Binance 为例）
exchange = ccxt.binance({
    'enableRateLimit': True,      # 自动限速
    # 如需测试网：'options': {'defaultType': 'future', 'fetchMarkets': 'https://testnet.binance.vision/...'}
})

# 2) 获取可交易对列表（基本信息）
markets = exchange.load_markets()

# # 比如筛出所有 USDT 计价的币对
# usdt_pairs = [symbol for symbol in markets if symbol.endswith('/USDT')]
# print('Total USDT pairs:', len(usdt_pairs))
#
# # 3) 拿单个交易对的最新价、24h 统计
# ticker = exchange.fetch_ticker('BTC/USDT')
# print('BTC/USDT price:', ticker['last'], '24h vol:', ticker['quoteVolume'])
#
# # 4) 获取历史 K 线 (OHLCV)，返回 [[ts,open,high,low,close,volume],…]
# #    timeframe 支持 '1m','5m','15m','1h','1d'…
# since = exchange.parse8601('2025-01-01T00:00:00Z')
# ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', since=since, limit=500)
# # 转成 DataFrame
# df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
# df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
# df = df.set_index('datetime').drop(columns=['ts'])
# print(df.head())
