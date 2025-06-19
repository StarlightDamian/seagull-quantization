# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:35:11 2025

@author: awei
"""



import pandas as pd
import numpy as np
def calculate_buy_sell_prices(day_df, buy_pct=[10, 20], sell_pct=[80, 90]):
    """
    计算Buy和Sell价格的百分位数
    :param day_df: 单日股票数据 (DataFrame)，需包含 'low', 'high', 'volume'
    :param buy_pct: 买入价格的百分位（例如10%买入）
    :param sell_pct: 卖出价格的百分位（例如80%卖出）
    :return: day_df，包含买入卖出价格以及VWAP
    """
    total_volume = day_df['volume'].sum()
    ohlc_pct_columns = []

    # 计算买入价格（低价百分位）
    day_df = day_df.sort_values('low')
    day_df['low_cum_volume'] = day_df['volume'].cumsum()
    
    for buy_pct_1 in buy_pct:
        buy_index = np.searchsorted(day_df['low_cum_volume'], (buy_pct_1 / 100) * total_volume)
        pct_low = f'_{buy_pct_1}pct_5min_low'
        day_df[pct_low] = day_df.iloc[buy_index]['low']
        ohlc_pct_columns.append(pct_low)
    
    # 计算卖出价格（高价百分位）
    day_df = day_df.sort_values('high', ascending=False)
    day_df['high_cum_volume'] = day_df['volume'].cumsum()
    
    for sell_pct_1 in sell_pct:
        sell_index = np.searchsorted(day_df['high_cum_volume'], (1 - (sell_pct_1 / 100)) * total_volume)
        pct_high = f'_{sell_pct_1}pct_5min_high'
        day_df[pct_high] = day_df.iloc[sell_index]['high']
        ohlc_pct_columns.append(pct_high)
    
    # 计算VWAP
    day_df['vwap'] = (((day_df['high'] + day_df['low'] + day_df['close']) / 3) * day_df['volume']).sum() / total_volume
    day_df['vwap'] = day_df['vwap'].round(4)

    # 返回最终计算的DataFrame，只保留相关列
    day_df = day_df[['vwap'] + ohlc_pct_columns]#'date', 'full_code', 
    return day_df

def process_daily_features_3d(daily_features_3d):
    """
    执行多只股票的数据处理并计算每个股票的次高价、次低价和VWAP。
    :param daily_features_3d: 一个DataFrame，包含多个股票（full_code），每个股票有每日（date）的OHLCV数据
    :return: 每只股票的计算结果，包含次高价、次低价和VWAP
    """
    result_list = []
    
    for symbol in daily_features_3d.columns.levels[1]:  # 遍历所有股票
        stock_data = daily_features_3d.xs(symbol, level=1, axis=1)  # 获取每只股票的数据
        stock_result = calculate_buy_sell_prices(stock_data)  # 计算每只股票的买卖价格和VWAP
        result_list.append(stock_result)
    
    # 合并所有股票的结果
    final_result = pd.concat(result_list, axis=0)
    return final_result


def add_max_high_column(daily_features_3d):
    """
    为 daily_features_3d 添加每个 stock_code 的最大 high 价格列。
    
    :param daily_features_3d: 原始的三维 DataFrame，带有 'date' 和 'stock_code' 的索引
    :return: 更新后的 DataFrame，新增 'max_high' 列
    """
    # 使用 groupby 对每个股票进行处理，找到每个股票在每个日期的最大 high
    max_high = daily_features_3d.groupby('stock_code')['high'].transform('max')
    
    # 将计算结果作为新的列加入到 daily_features_3d 中
    daily_features_3d['max_high'] = max_high
    
    return daily_features_3d

if __name__ == '__main__':
    # 假设我们有以下数据：
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
        'stock_code': ['AAPL', 'GOOG', 'AMZN', 'AAPL', 'GOOG'],
        'high': [150, 1200, 1900, 155, 1220],
        'low': [145, 1180, 1880, 150, 1200],
        'close': [148, 1195, 1890, 153, 1210],
        'volume': [1000000, 2000000, 1500000, 1100000, 2100000]
    }
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 设置 MultiIndex (date, stock_code)
    df.set_index(['date', 'stock_code'], inplace=True)
    
    # 查看数据
    print("DataFrame with MultiIndex:")
    print(df)
    
    # 计算每日的特征，例如：每日的最高价、最低价、收盘价和成交量
    daily_features = df.groupby(['date', 'stock_code']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # 查看结果
    print("\nDaily Features:")
    #print(daily_features)
    
    # 假设我们想要构建一个三维结构（date, stock_code, feature）
    # 将结果转换为三维数据结构，按照需要的格式
    daily_features.columns.name = 'parms'
    daily_features_3d = daily_features.unstack(level=1)
    
    # 查看结果
    print("\n3D Matrix-like structure:")
    #print(daily_features_3d)

        # 假设 daily_features_3d 现在是一个包含多个股票的数据
    #result_df = process_daily_features_3d(daily_features_3d)
    #print(result_df)
    
    daily_features_3d = add_max_high_column(daily_features_3d)

# =============================================================================
#     result_list = []
#     
#     for symbol in daily_features_3d.columns.levels[1]:  # 遍历所有股票
#         stock_data = daily_features_3d.xs(symbol, level=1, axis=1)  # 获取每只股票的数据
#         stock_result = calculate_buy_sell_prices(stock_data)  # 计算每只股票的买卖价格和VWAP
#         result_list.append(stock_result)
#     
#     # 合并所有股票的结果
#     final_result = pd.concat(result_list, axis=0)
# =============================================================================
