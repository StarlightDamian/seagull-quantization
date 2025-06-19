# -*- coding: utf-8 -*-
"""
@Date: 2023/8/22 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: crypto_incr_ccxt.py
@Description: digital_currency(ods/ohlc/crypto_incr_ccxt)
"""

import pandas as pd
import numpy as np
import ccxt
import time
import os
from datetime import timedelta, datetime
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import argparse
from utils.logger import setup_logger 
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--exchange", type=str, default="binance",
                    help="Name of the exchange")
parser.add_argument("--proxy", type=str, default="127.0.0.1:7890",
                    help="Proxy to use for the exchange")
parser.add_argument("--trade", type=int, default=0,
                    help="If this script will trade (use the apikey to trade) (default 0)")
parser.add_argument("--dataset-name", type=str, default="ohlcv",
                    help="The name of the dataset")
parser.add_argument("--symbol", type=str, default="BTC/USDT",
                    help="The symbol of the market (default: BTC/USDT)")
parser.add_argument("--time-interval", type=str, default="15m",
                    help="The tiem interval of the data (default: 15m)")


def init_exchange(exchange_name, proxy, need_apikey):
    exchange_function = getattr(ccxt, exchange_name)
    exchange = exchange_function({"enableRateLimit": True, "proxies": proxy})
    if need_apikey:
        apikey_path = os.path.join("apikeys", exchange_name, "key.txt")
        with open(apikey_path, "r") as f:
            keys = f.readlines()
        exchange.apiKey = keys[0].strip()
        exchange.secret = keys[1].strip()
        try:
            balance = exchange.fetch_balance()
            total_balance = {key: value for key, value in balance['total'].items() if value != 0}
        except ccxt.AuthenticationError:
            logging.error("Authentication error. Please check your API keys.")
            exit(1)
        logging.info("Set up exchange with apikey, Account balance: {}".format(total_balance))
        return exchange
    logging.info("Set up exchange without apikey")  
    return exchange

if __name__ == "__main__":
    args = parser.parse_args()
    log_path = setup_logger(log_type="get_dataset")
    logging.info("Using log path: {}".format(log_path))

    dataset_name = "{}-{}.csv".format(args.dataset_name, datetime.now().strftime("%Y-%m-%d")) 
    dataset_path = os.path.join("datasets", dataset_name)

    proxy = {"http": args.proxy, "https": args.proxy}
    logging.info("Using proxy:")
    logging.info(proxy)
    exchange_name = args.exchange
    exchange = init_exchange(exchange_name, proxy, args.trade)

    symbol = args.symbol
    time_interval = args.time_interval

    time_interval = '1d'
    since_time = datetime(2021, 1, 1)
    to_time = datetime(2024, 8, 1)
    logging.info("Getting data from {} to {}".format(since_time, to_time))

    df = pd.DataFrame() 
    while since_time < to_time:
        since = exchange.parse8601(since_time.strftime("%Y-%m-%d %H:%M:%S"))
        data = exchange.fetch_ohlcv(symbol=symbol, 
                                    timeframe=time_interval,
                                    since=since,
                                    limit=500)
        new_df = pd.DataFrame(data, dtype=float)
        new_df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
        new_df['candle_begin_time'] = pd.to_datetime(new_df['MTS'], unit='ms')
        new_df['candle_begin_time_GMT8'] = new_df['candle_begin_time'] + timedelta(hours=8)
        new_df = new_df[['candle_begin_time_GMT8', 'open', 'high', 'low', 'close', 'volume']]

        df = pd.concat([df, new_df], ignore_index=True)
        since_time = df['candle_begin_time_GMT8'].iloc[-1] + timedelta(days=1)

    df.to_csv(dataset_path, index=False)
    logging.info("Done.")


import ccxt
from datetime import datetime, timedelta
import pandas as pd
from seagull.settings import PATH
exchange = ccxt.binance({
    "enableRateLimit": True,
    "proxy": {
        "http": "127.0.0.1:7890",
        "https": "127.0.0.1:7890"
    }
})

exchange.apiKey = "your api key"
exchange.secret = "your secret key"

# test
balance = exchange.fetch_balance()
# 如果没有报错，说明apikey和secret正确

symbol = "BTC/USDT"
time_interval = '1d'
since_time = datetime(2021, 1, 1)
to_time = datetime(2024, 8, 1)

df = pd.DataFrame()
while since_time < to_time:
    # 将since_time转换为unix时间戳
    since = exchange.parse8601(since_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 获取数据, limit=500表示最多获取500条数据
    data = exchange.fetch_ohlcv(symbol=symbol,
                                timeframe=time_interval,
                                since=since,
                                limit=500)
    new_df = pd.DataFrame(data, dtype=float)
    new_df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
    # 把时间戳转换为datetime格式
    new_df['candle_begin_time'] = pd.to_datetime(new_df['MTS'], unit='ms')
    # 时差
    new_df['candle_begin_time_GMT8'] = new_df['candle_begin_time'] + timedelta(hours=8)
    new_df = new_df[['candle_begin_time_GMT8', 'open', 'high', 'low', 'close', 'volume']]

    df = pd.concat([df, new_df], ignore_index=True)
    since_time = df['candle_begin_time_GMT8'].iloc[-1] + timedelta(days=1)

df.to_csv(f'{PATH}/data/demo_ccxt.csv', index=False)
