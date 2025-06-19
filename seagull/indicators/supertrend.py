# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:45:24 2025

@author: awei
supertrend
"""
import os

import vectorbt as vbt
import pandas as pd
import talib
# import numpy as np
# import matplotlib.pyplot as plt

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')
import pandas as pd
import numpy as np
import talib

def calculate_supertrend(df, period=14, multiplier=3):
    """
    计算SuperTrend指标。
    
    :param df: 包含股票的OHLC数据（Open, High, Low, Close）
    :param period: ATR计算的周期，默认为14
    :param multiplier: 放大因子，控制灵敏度，默认为3
    :return: SuperTrend指示值
    """
    
    # 计算True Range (TR)
    #df['H-L'] = df['High'] - df['Low']
    #df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    #df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    #df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # 计算ATR
    #df['ATR'] = df['TR'].rolling(window=period).mean()
    df['atr'] = talib.ATR(df['high'], df['low'], df.close, timeperiod=14)
    # 计算UpperBand和LowerBand
    df['upper_basic'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
    df['lower_basic'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
    
    # 初始化SuperTrend（使用向量化方法来判断趋势方向）
    df['upper_band_prev'] = df['upper_basic'].shift(1)
    df['lower_band_prev'] = df['lower_basic'].shift(1)
    
    # 当前趋势计算（UpTrend或DownTrend）
    df['super_trend'] = np.where(df['close'] > df['upper_band_prev'], df['lower_basic'],
                                np.where(df['close'] < df['lower_band_prev'], df['upper_basic'], np.nan))
    
    # 填充空值，确保趋势保持
    df['super_trend'].ffill(inplace=True)
    
    return df['super_trend']
# =============================================================================
#     # 计算SuperTrend
#     df['UpperBasic'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
#     df['LowerBasic'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']
# 
#     # 初始化SuperTrend
#     df['UpperBand'] = df['UpperBasic']
#     df['LowerBand'] = df['LowerBasic']
#     df['SuperTrend'] = np.nan
# 
#     # 确定SuperTrend方向
#     for i in range(1, len(df)):
#         if df['Close'].iloc[i] > df['UpperBand'].iloc[i-1]:
#             df['SuperTrend'].iloc[i] = df['LowerBand'].iloc[i]  # 下跌趋势
#         elif df['Close'].iloc[i] < df['LowerBand'].iloc[i-1]:
#             df['SuperTrend'].iloc[i] = df['UpperBand'].iloc[i]  # 上涨趋势
#         else:
#             df['SuperTrend'].iloc[i] = df['SuperTrend'].iloc[i-1]  # 保持不变
# 
#     return df['SuperTrend']
# =============================================================================



if __name__ == '__main__':

    with utils_database.engine_conn("POSTGRES") as conn:
        raw_df = pd.read_sql("select primary_key, date,high,low, close from dwd_ohlc_incr_stock_daily where full_code='510300.sh'", con=conn.engine)
    
    raw_df = raw_df.drop_duplicates('primary_key', keep='first')
    raw_df.sort_values(by='date', ascending=True, inplace=True)
    
    # raw_df = raw_df[raw_df.date<'2024-10-08']
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    price = raw_df.set_index('date')['close']
    raw_df['super_trend'] = calculate_supertrend(raw_df)
    
    entries = price > raw_df['super_trend'].values  # RSI小于标准差，视为买入信号
    exits = price < raw_df['super_trend'].values  # RSI大于标准差，视为卖出信号
    portfolio = vbt.Portfolio.from_signals(price,
                                           entries,
                                           exits,
                                           freq='d',
                                           init_cash=10000,
                                           fees=0.001,
                                           slippage=0.001,
                                           )
    fig = portfolio.plot()  # .figure
    fig.write_html(f"{PATH}/html/portfolio_plot2.html")
    logger.info(portfolio.stats(settings=dict(freq='d',
                                              year_freq='243 days')))
    
    logger.info(portfolio.returns_stats(settings=dict(freq='d'),
                                        year_freq='243 days',))