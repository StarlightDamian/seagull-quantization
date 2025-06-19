# -*- coding: utf-8 -*-
"""
@Date: 2024/5/20 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: realtime_stock.py
@Description: 实时数据
"""
import efinance as ef
df = ef.stock.get_realtime_quotes()

df = df.rename(columns={'股票名称': 'code_name',
                        '股票代码': 'asset_code',
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'turnover',
                        '振幅': 'amplitude',  # new
                        '涨跌幅': 'chg_rel',
                        '涨跌额': 'price_chg',  # new
                        '换手率': 'turnover_pct',
                        })
df = df[df['asset_code'].isin(['600418', '516010', '300548'])]
print(df.T)

