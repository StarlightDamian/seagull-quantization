# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:33:30 2024

@author: awei
(data_ods_info_full_asset_base)
"""
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

df.to_csv(dataset_path, index=False)