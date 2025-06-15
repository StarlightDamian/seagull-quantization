# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:13:24 2024

@author: awei
backtest_vectorbt_signal
"""

import vectorbt as vbt
import pandas as pd

def backtest_vectorbt(portfolio):
    #backtest_dict = dict(portfolio.stats())
    backtest_dict={'total_return':portfolio.total_return(),
                   'sharpe_ratio': portfolio.sharpe_ratio()}
    return backtest_dict

def pipline(symbols):
    strategy_results = {}
    base_results = {}
    
    total_return_list = []
    sharpe_ratio_list = []
    for symbol in symbols:
        # Download data for the individual symbol
        data = vbt.YFData.download(symbol, start='2021-01-01', end='2022-01-01', missing_index='drop').get('Close')
        
        # Define simple moving average (SMA)
        fast_ma = vbt.MA.run(data, window=10)
        slow_ma = vbt.MA.run(data, window=50)

        # Generate buy and sell signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Create portfolios
        strategy_portfolio = vbt.Portfolio.from_signals(data, entries, exits, init_cash=10000, freq='1D')
                
        total_return = strategy_portfolio.total_return()
        sharpe_ratio = strategy_portfolio.sharpe_ratio()

        total_return_list.append(total_return)
        sharpe_ratio_list.append(sharpe_ratio)
        
        
    # Combine results into a DataFrame
    results = pd.DataFrame({
        'Total Return': total_return_list,
        'Sharpe Ratio': sharpe_ratio_list
    })
    print(results)
        #base_portfolio = vbt.Portfolio.from_holding(data, init_cash=10000, freq='1D')

        # Store individual results in the dictionary
        #strategy_results[symbol] = backtest_vectorbt(strategy_portfolio)
        #base_results[symbol] = backtest_vectorbt(base_portfolio)

    #return base_results, strategy_results

if __name__ == '__main__':
    # List of symbols
    symbols = ['AAPL', 'MSFT']#, 'GOOG'
    #base_dicts, strategy_dicts = pipline(symbols)
    
    strategy_results = {}
    base_results = {}
    total_return_list = []
    sharpe_ratio_list = []
    for symbol in symbols:
        # Download data for the individual symbol
        data = vbt.YFData.download(symbol, start='2021-01-01', end='2022-01-01', missing_index='drop').get('Close')
        
        # Define simple moving average (SMA)
        fast = vbt.MA.run(data, window=10)
        slow = vbt.MA.run(data, window=50)

        # Generate buy and sell signals
        entries = fast.ma_crossed_above(slow)
        exits = fast.ma_crossed_below(slow)
        
        # Create portfolios
        strategy_portfolio = vbt.Portfolio.from_signals(data, entries, exits, init_cash=10000, fees=0.001, freq='1D')
        #base_portfolio = vbt.Portfolio.from_holding(data, init_cash=10000, freq='1D')
        
        total_return = strategy_portfolio.total_return()
        sharpe_ratio = strategy_portfolio.sharpe_ratio()

        total_return_list.append(total_return)
        sharpe_ratio_list.append(sharpe_ratio)
        
        
    # Combine results into a DataFrame
    results = pd.DataFrame({
        'Total Return': total_return_list,
        'Sharpe Ratio': sharpe_ratio_list
    })
    print(results)
# =============================================================================
#         # Store individual results in the dictionary
#         strategy_results[symbol] = backtest_vectorbt(strategy_portfolio)
#         base_results[symbol] = backtest_vectorbt(base_portfolio)
#     print(base_results)
#     print(strategy_results)
# =============================================================================
    
    # Print results for each symbol
# =============================================================================
#     for symbol in symbols:
#         print(f"Base portfolio stats for {symbol}: {base_dicts[symbol]}")
#         print(f"Strategy portfolio stats for {symbol}: {strategy_dicts[symbol]}")
# =============================================================================
        
        
# =============================================================================
#     Base portfolio stats for AAPL: {'total_return': 0.34648191490511415, 'sharpe_ratio': 1.5744591471730254}
#     Strategy portfolio stats for AAPL: {'total_return': 0.2410581943762765, 'sharpe_ratio': 1.6362927333367396}
#     Base portfolio stats for MSFT: {'total_return': 0.5247692895262035, 'sharpe_ratio': 2.534883727673892}
#     Strategy portfolio stats for MSFT: {'total_return': 0.17103921645697537, 'sharpe_ratio': 1.3206492280059032}
#     Base portfolio stats for GOOG: {'total_return': 0.6517056791852789, 'sharpe_ratio': 2.68539338690476}
#     Strategy portfolio stats for GOOG: {'total_return': 0.022825258398281402, 'sharpe_ratio': 0.3222488512997241}
# =============================================================================
