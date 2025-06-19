# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:30:07 2024

@author: awei
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from __init__ import path
from utils import utils_log
from backtest import vectorbt_base

class backtestVectorbtMacd(vectorbt_base.backtestVectorbt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macd_hist = None
        
    def strategy_params(self, strategy_params_list):
        strategy_params_df = pd.DataFrame(strategy_params_list)
        strategy_params_df['primary_key'] = strategy_params_df['window_fast'].astype(str)+'-'+ \
                                            strategy_params_df['window_slow'].astype(str)+'-'+ \
                                            strategy_params_df['window_signal'].astype(str)
        return strategy_params_df
    
    def strategy_base(self, subtable_df, data):
        window_fast, window_slow, window_signal = subtable_df[['window_fast', 'window_slow', 'window_signal']].values[0]
    
        # https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/indicators/nb.py#L171
        macd = vbt.MACD.run(
            close=data,  # close: 2D数组，表示收盘价
            fast_window=window_fast,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
            slow_window=window_slow,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
            signal_window=window_signal,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
            macd_ewm=False,  # #布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
            signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
            adjust=False #布尔值，是否在计算EMA时进行调整
            #cache_dict,字典，用于缓存计算结果
        )
        
        #self.macd_hist = macd.hist # macd能量柱
        return macd

if __name__ == '__main__':
    symbols = ["AAPL",'MSFT']#, "AAPL"
    price = vbt.YFData.download(symbols, missing_index='drop').get('Close')
    
    # 策略参数
    strategy_params_list = [
        {'window_fast': window_fast,  # Fast EMA period, Default value 12
         'window_slow': window_slow,  # Slow EMA period, Default value 26
         'window_signal':window_signal,  # Signal line period, Default value 9
         }
        for window_fast, window_slow, window_signal in itertools.product(
            list(range(10, 15)),
            list(range(24,29)),
            list(range(7,11))) if window_slow > window_fast
        ]
    
    backtest_vectorbt_macd = backtestVectorbtMacd(output='database',
                                                  use_multiprocess=False,  # [False, True]
                                                  output_trade_details=False,
                                                  strategy_params_batch_size=512,  # MemoryError
                                                  portfolio_params={'freq': 'd',
                                                                    'fees': 0.001,  # 0.1% per trade
                                                                    'slippage': 0.001,  # 0.1% slippage
                                                                    'init_cash': 10000},
                                                  strategy_params_list=strategy_params_list,
                                                  )
    
    backtest_vectorbt_macd.ablation_experiment(date_start='2019-01-01', 
                                               date_end='2023-01-01',
                                               comparison_experiment="macd_diff_20241010_6",
                                               )
    