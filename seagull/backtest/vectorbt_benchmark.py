# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:51:28 2024

@author: awei
backtest_vectorbt_base
"""

import vectorbt as vbt

from __init__ import path
from backtest_vectorbt import backtestVectorbt


class backtestVectorbtAlpha(backtestVectorbt):
    def __init__(self):
        super().__init__()

    def sh_510300():
        # base: 沪深300
        # Compare against benchmark (e.g., CSI 300)
        benchmark_data = vbt.YFData.download('510300.SH', start='2018-01-01', end='2022-01-01').get('Close')
        benchmark_portfolio = vbt.Portfolio.from_holding(benchmark_data, init_cash=10000)
        benchmark_stats = benchmark_portfolio.stats()
        
        # Print benchmark performance
        print(f"Benchmark Total Return: {benchmark_stats['Total Return [%]']}")
        print(f"Benchmark Sharpe Ratio: {benchmark_stats['Sharpe Ratio']}")
        
if __name__ == '__main__':
    benchmark_data = vbt.YFData.download('SPX.USI', start='2018-01-01', end='2022-01-01')