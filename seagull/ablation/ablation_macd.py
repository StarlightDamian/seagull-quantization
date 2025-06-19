# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from seagull.settings import PATH
from seagull.utils import utils_log
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class ablationMacd(vectorbt_macd.backtestVectorbtMacd, analyze.backtestAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macd_hist = None
    
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
        
        self.macd_hist = macd.hist # macd能量柱
        

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
    ablation_macd = ablationMacd(output='database',
                                                  use_multiprocess=False,  # [False, True]
                                                  output_trade_details=False,
                                                  strategy_params_batch_size=512,  # MemoryError
                                                  portfolio_params={'freq': 'd',
                                                                    'fees': 0.001,  # 0.1% per trade
                                                                    'slippage': 0.001,  # 0.1% slippage
                                                                    'init_cash': 10000},
                                                  strategy_params_list=strategy_params_list,
                                                  )
    comparison_experiment = "macd_diff_20241011_1"
    ablation_macd.ablation_experiment(date_start='2019-01-01', 
                                               date_end='2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
# =============================================================================
#     2024-10-11 20:31:11.093 | INFO     | backtest.analyze:pipeline:163 - score_mean_base: 20.657
#     2024-10-11 20:31:11.109 | INFO     | backtest.analyze:pipeline:164 - score_mean_strategy_effective: 25.382
#     2024-10-11 20:31:11.109 | INFO     | backtest.analyze:pipeline:175 - strategy_rank: 
#     window
#     13-28-7           32.130
#     13-25-9           31.573
#     13-26-8           31.573
#     13-25-7           31.459
#     11-26-7           31.392
#      
#     12-24-10          24.953
#     11-24-10          24.808
#     10-24-7           24.739
#     10-24-8           24.189
#     None-None-None    20.657
#     Length: 37, dtype: float64
#     2024-10-11 20:31:11.187 | INFO     | backtest.analyze:pipeline:179 - rank_portfolio: 
#     full_code
#     SH.515700    42.572
#     SH.515030    41.734
#     SH.512580    40.411
#     SZ.159949    40.155
#     SH.512690    39.379
#      
#     SZ.159954    12.314
#     SH.512200    12.220
#     SZ.159940    12.215
#     SH.510230    10.618
#     SZ.159931    10.290
#     Length: 688, dtype: float64
#     2024-10-11 20:31:11.187 | INFO     | backtest.analyze:pipeline:187 - baseline: 26.089
#     2024-10-11 20:31:11.187 | INFO     | backtest.analyze:pipeline:188 - baseline_strategy: 28.576
#     2024-10-11 20:31:11.296 | INFO     | backtest.analyze:pipeline:192 - comparison_portfolio: 
#                     base   strategy  strategy_better
#     full_code                                       
#     SH.513060   3.796000  27.262000         23.46600
#     SH.513980   3.595000  26.207000         22.61200
#     SH.513860   3.774000  25.633000         21.85900
#     SH.513360   2.781000  22.505000         19.72400
#     SH.517350   8.913000  27.940000         19.02700
#                  ...        ...              ...
#     SH.510210  32.546000  19.242000        -13.30400
#     SH.512390  34.175000  19.458000        -14.71700
#     SH.512040  37.158000  17.306000        -19.85200
#     SH.512690  63.031000  38.433000        -24.59800
#     mean       20.402077  25.226487          4.82441
# 
#     [664 rows x 3 columns]
#     2024-10-11 20:31:11.312 | INFO     | backtest.analyze:pipeline:196 - rank_personal: 
#            full_code comparison_experiment          window  bar_num  ann_return  \
#     110    SH.512690                  base  None-None-None     1335      69.058   
#     72553  SH.512690    macd_diff_20241011        14-24-10      892      44.809   
#     20189  SH.512690    macd_diff_20241011        10-25-10      892      43.911   
#     61655  SH.515700    macd_diff_20241011        13-25-10      707      35.619   
#     56088  SH.515030    macd_diff_20241011        12-28-10      690      34.638   
#              ...                   ...             ...      ...         ...   
#     512    SZ.159740                  base  None-None-None      583     -14.023   
#     143    SH.513130                  base  None-None-None      578     -14.456   
#     517    SZ.159747                  base  None-None-None      535     -14.873   
#     157    SH.513360                  base  None-None-None      562     -12.551   
#     156    SH.513330                  base  None-None-None      691     -17.584   
# 
#            max_dd   score  
#     110    49.451  63.031  
#     72553  36.630  50.179  
#     20189  41.536  48.304  
#     61655  25.232  47.427  
#     56088  25.091  46.791  
#           ...     ...  
#     512    62.231   3.364  
#     143    62.330   3.019  
#     517    61.561   2.934  
#     157    68.663   2.781  
#     156    67.446  -0.492  
# 
#     [17170 rows x 7 columns]
# =============================================================================
    
    
    