# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:44:35 2024

@author: awei
finance_limit
"""
import pandas as pd
import numpy as np

def limit_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算股票的涨停价和跌停价，并进行取整。
    
    :param df: 包含两列的DataFrame，分别为 'close'（收盘价）和 'price_limit_rate'（涨跌幅）。
    :return: 返回一个DataFrame，包含涨停价和跌停价。
    目前ETF和北交所没有'prev_close'
    """
    # df = df.sort_values(by='date', ascending=True)
    
    # 计算涨停价和跌停价
    limit_up = df['prev_close'] * (1 + df['price_limit_rate'])  # 涨停价格
    limit_down = df['prev_close'] * (1 - df['price_limit_rate'])  # 跌停价格
    
    # 涨跌停价格
    df['limit_up'] = np.floor(limit_up * 100) / 100  # 涨停价向上取整到小数点后2位
    df['limit_down'] = np.ceil(limit_down * 100) / 100  # 跌停价向下取整到小数点后2位
    
    df['is_limit_up'] = np.where(df['high'] >= df['limit_up'], True, False)
    df['is_limit_down'] = np.where(df['low'] <= df['limit_down'], True, False)
    
    # 一字板
    df['is_flat_price'] = np.where(df['high'] == df['low'], True, False)
    
    df[['is_limit_up_prev','is_limit_down_prev']] = df[['is_limit_up','is_limit_down']].shift(1)
    
    # 目前看异常数据意义不大，即使是超出涨跌停，大部分也不会超出太多
    # abnormal = ((df['high']>df['limit_up']*1.01)|(df['low']<df['limit_down']*0.99))
    # df.loc[abnormal,'is_abnormal'] = True  # 异常数据
    # df.loc[~abnormal,'is_abnormal'] = False
    
    return df


if __name__ == '__main__':
    # 示例：假设有5000只股票的收盘价和对应的涨跌幅
    stock_codes = [f'Stock_{i}' for i in range(1, 5001)]
    # 收盘价：10.00, 15.00, 20.00, ..., 100.00
    closing_prices = pd.Series([10.0 + i * 0.05 for i in range(5000)], index=stock_codes)
    # 涨跌幅：0.1, 0.2, 0.3, ..., 0.1 (例如，有的板块涨跌幅是10%，有的板块是20%)
    price_limit_rate = pd.Series([0.1 + (i % 3) * 0.1 for i in range(5000)], index=stock_codes)
    
    # 将收盘价和涨跌幅组合成一个DataFrame
    df = pd.DataFrame({
        'close': closing_prices,
        'price_limit_rate': price_limit_rate
    })
    
    # 计算涨停价和跌停价
    limit_prices_df = limit_prices(df)
    
    # 输出结果
    print(limit_prices_df.head())
    