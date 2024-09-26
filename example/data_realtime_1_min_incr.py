# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:55:14 2024

@author: awei
截面数据
最新数据(data_realtime_1_min)
无ETF
columns=['股票代码', '股票名称', '涨跌幅', '最新价', '最高', '最低', '今开', '涨跌额', '换手率', '量比',
       '动态市盈率', '成交量', '成交额', '昨日收盘', '总市值', '流通市值', '行情ID', '市场类型', '更新时间',
       '最新交易日']
"""
import pandas as pd
import efinance as ef
df = ef.stock.get_realtime_quotes()
df = df.rename(columns={'股票代码': 'code',
                        '股票名称': 'code_name',
                        '涨跌幅': 'pct_chg',
                        '最新价': 'latest_price',
                        '最高': 'high',
                        '最低': 'low',
                        '今开': 'open',
                        '涨跌额': 'price_chg',
                        '换手率': 'turnover_rate',
                        '量比': 'volume_ratio',
                        '动态市盈率': 'pe_ttm',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '昨日收盘': 'pre_close',
                        '总市值': 'total_market_cap',
                        '流通市值': 'circulating_market_cap',
                        '行情ID': 'market_id',
                        '市场类型': 'market_type',
                        '更新时间': 'update_time',
                        '最新交易日': 'date',
                        })
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df = df.sort_values(by='amount',ascending=False)
result_df = df.sort_values(by='amount',ascending=False).head(20)[['code','code_name','amount','pct_chg','price_chg','pe_ttm']]
print(result_df)

