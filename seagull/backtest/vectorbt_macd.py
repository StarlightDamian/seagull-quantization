# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:01:17 2024

@author: awei
(backtest_vectorbt_macd)

macd特征
1.适合趋势，不适合震荡
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from __init__ import path
from utils import log
from vectorbt_base import backtestVectorbt

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = log.logger_config_local(f'{path}/log/{log_filename}.log')


class backtestVectorbtMacd(backtestVectorbt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def strategy_params(self, strategy_params_list):
        strategy_params_df = pd.DataFrame(strategy_params_list)
        strategy_params_df['primary_key'] = strategy_params_df['window_fast'].astype(str)+'-'+ \
                                            strategy_params_df['window_slow'].astype(str)+'-'+ \
                                            strategy_params_df['window_signal'].astype(str)
        return strategy_params_df
    
    def strategy(self, subtable_df, data):
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
        # cache_dict 示例数据
        # 用于存储预先计算的移动平均线，以提高性能
        # 创建缓存字典（这里只是示例，实际使用时需要预先计算）
        # cache_dict = {
        #     hash((fast_window, macd_ewm)): np.random.randn(100, 2),  # 模拟快速MA
        #     hash((slow_window, macd_ewm)): np.random.randn(100, 2)   # 模拟慢速MA
        # }
        
        # https://github.com/polakowo/vectorbt/issues/136
        entries = macd.macd_above(0) & macd.macd_below(0).vbt.signals.fshift(1)
        exits = macd.macd_below(0) & macd.macd_above(0).vbt.signals.fshift(1)
        
        entries_exits_t = pd.concat([entries, exits], axis=1, keys=['entries', 'exits']).T
        return entries_exits_t
        
    def ablation_experiment(self, symbols=[],
                            date_start='2020-01-01',
                            date_end='2022-01-01',
                            comparison_experiment=None,
                            if_exists='fail',  # ['fail','replace','append']
                            ):
        # dataset
        portfolio_df = self.dataset(symbols, date_start=date_start, date_end=date_end)
        
        # base
        self.backtest(data=portfolio_df)
        
        # strategy
        self.backtest(portfolio_df, comparison_experiment=comparison_experiment)
        
        
if __name__ == '__main__':
    # 策略参数
    strategy_params_list = [
        {'window_fast': window_fast,  # Fast EMA period, Default value 12
         'window_slow': window_slow,  # Slow EMA period, Default value 26
         'window_signal':window_signal,  # Signal line period, Default value 9
         }
        for window_fast, window_slow, window_signal in itertools.product(list(range(10, 15)),
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
                                               comparison_experiment="macd_20240925",
                                               )
    