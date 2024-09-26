# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:59:59 2024

@author: awei
volatility_signal
适合震荡期交易的特征和信号

a) RSI (Relative Strength Index)：

- 超买（通常RSI > 70）考虑卖出
- 超卖（通常RSI < 30）考虑买入

b) 布林带（Bollinger Bands）：

- 价格接近上轨考虑卖出
- 价格接近下轨考虑买入

c) 随机指标（Stochastic Oscillator）：

- 超买区（通常K% > 80）考虑卖出
- 超卖区（通常K% < 20）考虑买入

d) 价格动量：

- 短期均线与长期均线的交叉
"""

import vectorbt as vbt
import numpy as np

def oscillation_trading_strategy(symbol, start_date, end_date, rsi_window=14, bb_window=20, bb_std=2):
    # Fetch data
    data = vbt.YFData.download(symbol, start=start_date, end=end_date)
    close = data.close
    
    # Calculate RSI
    rsi = vbt.indicators.RSI.run(close, window=rsi_window).rsi.to_numpy()
    
    # Calculate Bollinger Bands
    bb = vbt.indicators.BB.run(close, window=bb_window, sigma=bb_std)
    
    # Generate entry signals
    buy_signal = (rsi < 30) & (close < bb.lower)
    sell_signal = (rsi > 70) & (close > bb.upper)
    
    # Run the portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=buy_signal,
        exits=sell_signal,
        init_cash=10000,
        fees=0.001
    )
    
    # Calculate performance metrics
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    # Plot results
    fig = vbt.plotting.Figure()
    fig.add_candlestick(data.open, data.high, data.low, data.close, name='Price')
    fig.add_scatter(x=close.index, y=bb.middle, name='BB Middle')
    fig.add_scatter(x=close.index, y=bb.upper, name='BB Upper')
    fig.add_scatter(x=close.index, y=bb.lower, name='BB Lower')
    fig.add_scatter(x=close.index[buy_signal], y=close[buy_signal], name='Buy Signal', marker='triangle-up', color='green')
    fig.add_scatter(x=close.index[sell_signal], y=close[sell_signal], name='Sell Signal', marker='triangle-down', color='red')
    fig.add_scatter(x=close.index, y=rsi, name='RSI', yaxis='y2')
    fig.update_layout(title=f'Oscillation Trading Strategy for {symbol}',
                      yaxis2=dict(overlaying='y', side='right', range=[0, 100]))
    fig.show()

# Run the strategy
oscillation_trading_strategy('AAPL', '2020-01-01', '2023-12-31')