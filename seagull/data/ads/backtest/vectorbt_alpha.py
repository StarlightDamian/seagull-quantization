# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:54:41 2024

@author: awei
backtest_vectorbt_alpha
"""
import vectorbt as vbt

from seagull.settings import PATH
from backtest_vectorbt import backtestVectorbt


class backtestVectorbtAlpha(backtestVectorbt):
    def __init__(self):
        super().__init__()
        
    def month(self, portfolio):
        portfolio.returns().resample('M').sum()
        
    def backtest_metrics(self, portfolio, freq='210d'): 
        ...
        
    def reshape(self, data, freq='d'):
        return data.resample(freq).last()  # freq in ['ME','20D','Y','W']
    
    def data:
        # Acquire historical market data
        def get_market_data(symbols, start_date, end_date):
            data = yf.download(symbols, start=start_date, end=end_date)
            return data['Adj Close']
        
        # Acquire benchmark (e.g., CSI 300)
        def get_benchmark_data(benchmark_symbol, start_date, end_date):
            benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)
            return benchmark_data['Adj Close']
        
    def Preprocess(data):
        data = data.fillna(method='ffill')
        returns = data.pct_change().dropna()
        return returns
    
    
if __name__ == '__main__':
    data = vbt.YFData.download('BTC-USD', start='2018-01-01', end='2022-01-01', missing_index='drop').get('Close')
    
    
    strategy_portfolio.stats(settings=dict(freq='20d'),agg_func=None)
    
    
    
# =============================================================================
#        strategy_max_drawdown  strategy_annual_return  index_return
# 2018                -12.50%                 7.23%           -25.00%
# 2019                -10.35%                14.11%           36.07%
# 2020                 -8.21%                21.67%           27.20%
# 2021                -15.48%                 9.89%           8.60%
# 
# =============================================================================
