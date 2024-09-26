# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:20:18 2024

@author: awei
adx

ADX的计算步骤：

1.计算 +DM 和 -DM（Directional Movement）
2.计算 TR（True Range）
3.计算 +DI 和 -DI（Directional Indicator）
4.计算 DX（Directional Index）
5.计算 ADX

这个示例代码展示了如何使用vectorbt计算ADX指标，并基于ADX创建一个简单的交易策略。
这个策略在ADX大于25且+DI大于-DI时入场，在ADX大于25且+DI小于-DI时出场。
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

def calculate_adx(high, low, close, period=14):
    # Calculate +DM, -DM, and TR
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = vbt.indicators.ATR.run(high, low, close, window=1).atr.to_numpy()

    # Calculate smoothed +DM, -DM, and TR
    smoothed_plus_dm = vbt.generic.nb.ewm_mean_nb(plus_dm, period)
    smoothed_minus_dm = vbt.generic.nb.ewm_mean_nb(minus_dm, period)
    smoothed_tr = vbt.generic.nb.ewm_mean_nb(tr, period)

    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX
    adx = vbt.generic.nb.ewm_mean_nb(dx, period)

    return pd.DataFrame({
        '+DI': plus_di,
        '-DI': minus_di,
        'ADX': adx
    }, index=close.index)

# Example usage
def run_adx_strategy(symbol, start_date, end_date):
    # Fetch data
    data = vbt.YFData.download(symbol, start=start_date, end=end_date)

    # Calculate ADX
    adx_df = calculate_adx(data.high, data.low, data.close)

    # Create entry and exit signals
    entry_signal = (adx_df['ADX'] > 25) & (adx_df['+DI'] > adx_df['-DI'])
    exit_signal = (adx_df['ADX'] > 25) & (adx_df['+DI'] < adx_df['-DI'])

    # Run the portfolio
    portfolio = vbt.Portfolio.from_signals(
        data.close,
        entries=entry_signal,
        exits=exit_signal,
        init_cash=10000,
        fees=0.001
    )

    # Get performance metrics
    metrics = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()

    print(f"Total Return: {metrics:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot the results
    portfolio.plot().show()

# Run the strategy
run_adx_strategy('AAPL', '2020-01-01', '2023-12-31')