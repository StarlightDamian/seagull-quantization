# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:47:08 2024

@author: awei
dwd_feat_incr_indicators
demo_talib
https://ta-lib.org/functions/

TA-Lib 提供了非常丰富的技术分析指标，涵盖了趋势、动量、波动性、量能等多个方面。在实际使用中，你可以根据策略的需求选择合适的指标。上述代码涵盖了常见的技术指标，且在很多量化交易中应用广泛。通过这些指标，你可以构建包括超买超卖、趋势反转、动量指标等在内的策略。
"""
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
#import sys
#sys.path.append('D:/Program Files/TA-Lib')

import talib
import pandas as pd

from __init__ import path
from utils import utils_database, utils_data
from finance import finance_trading_day

def indicators(df):
    """
    计算技术指标，输入为某只股票的历史数据
    :param df: 每个股票的 DataFrame
    :return: 含有技术指标的 DataFrame
    """
    # 获取必要的列
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    # open_ = df['open']

    # 布林带（Bollinger Bands, BBANDS）: 通过计算价格的标准差，判断价格的波动范围，常用于判断市场的过度买卖。
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 相对强弱指数（RSI）:衡量股票在一定时间内的涨跌幅度，常用来判断市场是否超买或超卖。
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # 随机指标（Stochastic Oscillator, KDJ）:通过比较当前价格和过去一段时间的价格区间来评估超买超卖情况，通常用于判断趋势反转。
    fastk, fastd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

    # 商品通道指数（CCI）:通过比较价格与其移动平均线的偏离程度，衡量市场的超买超卖状态。
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)

    # 威廉指标（Williams %R, WR）:用于衡量股票超买超卖情况，类似于随机指标（KDJ），但它是反向的。
    df['wr'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # 成交量加权平均价（VWAP）: 结合成交量与价格计算的加权平均价格，用于评估某段时间内的价格走势。
    df['vwap'] = talib.SMA(close * volume, timeperiod=20) / talib.SMA(volume, timeperiod=20)
    
    # 积累/分配线（A/D Line, AD）: 通过考虑价格变动与成交量的关系来衡量市场的买入或卖出压力。
    df['ad'] = talib.AD(high, low, close, volume)
    
    # 动量（Momentum, MOM）:衡量价格变动的速度，用于判断趋势的强度。
    df['mom'] = talib.MOM(close, timeperiod=10)

    # 平均真实范围（ATR）:衡量市场波动性的一个指标，常用于计算止损位。
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)

    # 陀螺仪指标（ADX, Average Directional Index）:用于测量市场趋势的强度，常与 +DI 和 -DI 一起使用。
    # adx, plus_di, minus_di = talib.ADX(high, low, close, timeperiod=14)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    
    # 资金流向（Money Flow Index, MFI）: 结合价格和成交量，衡量市场的资金流入和流出，类似于 RSI，但考虑了成交量。
    df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

    # 分型指标（Fractal）:用于检测价格的反转点，常用于技术分析中的支撑和阻力位。
    # df['fractal_up'] = talib.FRACTALUP(close)  # bug:AttributeError: module 'talib' has no attribute 'FRACTALUP'
    # df['fractal_down'] = talib.FRACTALDOWN(close)

    # 价格通道（Price Channel）:常用于判断趋势的强度和市场的过度买卖状态。
    # pc_up, pc_down = talib.PRICECHANNEL(high, low, close, timeperiod=20)

# =============================================================================
#     # 计算移动平均线（例如 50 日和 200 日）
#     df['ma50'] = talib.SMA(df['close'], timeperiod=50)
#     df['ma200'] = talib.SMA(df['close'], timeperiod=200)
#     
#     # 计算收盘价相对于50日均线的强度（示例）
#     df['rsrs'] = (df['close'] - df['ma50']) / df['ma200']  # 收盘价与50日均线的差值
# =============================================================================
    
    # 将计算结果添加到原始数据中
    df['upper_band'] = upper
    df['middle_band'] = middle
    df['lower_band'] = lower
    df['kdj_fastk'] = fastk
    df['kdj_fastd'] = fastd
    # df['pc_up'] = pc_up
    # df['pc_down'] = pc_down
    return df

def pipeline(date_start, date_end):
    with utils_database.engine_conn('postgre') as conn:
        df = pd.read_sql(f"""
                         SELECT
                             primary_key
                             ,date
                             ,full_code
                             ,high
                             ,low
                             ,close
                             ,volume
                         FROM
                             dwd_ohlc_incr_stock_daily
                         where
                             date between '{date_start}' and '{date_end}'
                         """, con=conn.engine)
        
    #indicators_df = df.groupby('full_code').apply(indicators)
    grouped = df.groupby('full_code')
    # 使用 ThreadPoolExecutor 来并行处理每个股票的数据
    with ThreadPoolExecutor(max_workers=8) as executor:
        indicators_full_code_df = list(executor.map(indicators, [group for _, group in grouped]))
    
    # 将结果合并回一个 DataFrame
    indicators_df = pd.concat(indicators_full_code_df)
    return indicators_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='1990-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='', help='End time for backtesting')
    parser.add_argument('--update_type', type=str, default='full', help='Data update method')
    parser.add_argument('--filename', type=str, default='dwd_feat_incr_indicators', help='Database table name')
    args = parser.parse_args()
    
    date_end = args.date_end if args.date_end!='' else datetime.now().strftime("%F")
    if args.update_type=='full':
        df = pipeline(date_start=args.date_start,
                      date_end=date_end)
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='replace')
    elif args.update_type=='incr':
        date_start = utils_data.maximum_date_next(table_name=args.filename)
        trading_day_alignment = finance_trading_day.TradingDayAlignment()
        date_start_prev = trading_day_alignment.shift_day(date_start=date_start, date_num=200)
        
        raw_df = pipeline(date_start=date_start_prev, date_end=date_end)
        df = raw_df[raw_df.date>=date_start]
        
        utils_data.output_database_large(df,
                                         filename=args.filename,
                                         if_exists='append')
        