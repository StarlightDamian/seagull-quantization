# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:04:27 2024

@author: awei
vectorbt_limit_order
限价单交易
"""

import vectorbt as vbt
import pandas as pd
import numpy as np

def price_level_trading_strategy(symbol, date_start, date_end, buy_threshold, sell_threshold):
    # Fetch data
    #data = vbt.YFData.download(symbol, start=start_date, end=end_date)
    #close = data.close
    data = vbt.YFData.download(symbol , start=date_start, end=date_end, missing_index='drop')
    close = data.get('Close')
    # Calculate daily buy and sell prices
    buy_price = close * (1 - buy_threshold)
    sell_price = close * (1 + sell_threshold)
    
    # Generate entry and exit signals
    entries = data.get('Low') <= buy_price
    exits = data.get('High') >= sell_price
    
    # Run the portfolio simulation
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=10000,
        fees=0.001,
        open=data.get('Open'),
        high=data.get('High'),
        low=data.get('Low'),
        #upon_opposite_entry='ExitAndEntry',  # Exit position when opposite signal occurs
        upon_long_conflict='Ignore',  # Ignore long entries when already in a long position
        upon_short_conflict='Ignore',  # Ignore short entries when already in a short position
        freq= 'd',
    )
    
# =============================================================================
#     # Plot results
#     fig = vbt.plotting.Figure()
#     fig.add_candlestick(data.Open, data.get('High'),data.get('Low'),data.get('Close'), name='Price')
#     fig.add_scatter(x=close.index, y=buy_price, name='Buy Price', line=dict(color='green', dash='dash'))
#     fig.add_scatter(x=close.index, y=sell_price, name='Sell Price', line=dict(color='red', dash='dash'))
#     fig.add_scatter(x=entries.index[entries], y=buy_price[entries], name='Buy Signal', marker='triangle-up', color='green')
#     fig.add_scatter(x=exits.index[exits], y=sell_price[exits], name='Sell Signal', marker='triangle-down', color='red')
#     fig.update_layout(title=f'Price Level Trading Strategy for {symbol}')
#     fig.show()
# =============================================================================
    
    # Return the portfolio object for further analysis if needed
    return portfolio

# Run the strategy
symbol = 'AAPL'
date_start = '2020-01-01'
date_end = '2023-12-31'
buy_threshold = 0.02  # Buy when price is 2% below the previous close
sell_threshold = 0.02  # Sell when price is 2% above the previous close

portfolio = price_level_trading_strategy(symbol, date_start, date_end, buy_threshold, sell_threshold)

# You can perform additional analysis on the portfolio object if needed
# For example, print the trade statistics
print(portfolio.trades.stats())