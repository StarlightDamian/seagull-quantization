# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from __init__ import path
from utils import utils_log, utils_character
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


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
        
        # 计算 DIF 和 DEA 的斜率
        dif_slope = macd.macd.diff()  # DIF 线的斜率
        
        # 交易信号
        entries = (dif_slope > 0) & (dif_slope.shift(1) <= 0)  # 买入条件：DIF 斜率为正
        exits = (dif_slope < 0) & (dif_slope.shift(1) >= 0)  # 卖出条件：DIF 斜率为负
        
        entries_exits_t = pd.concat([entries, exits], axis=1, keys=['entries', 'exits']).T
        return entries_exits_t
        
        
if __name__ == '__main__':
    # 策略参数
    strategy_params_list = [
        {'window_fast': window_fast,  # Fast EMA period, Default value 12
         'window_slow': window_slow,  # Slow EMA period, Default value 26
         'window_signal':window_signal,  # Signal line period, Default value 9
         }
        for window_fast, window_slow, window_signal in itertools.product(list(range(12, 13)),
                                                                         list(range(26,27)),
                                                                         list(range(9,10))) if window_slow > window_fast
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
    
    comparison_experiment = utils_character.generate_random_string(6)  # "ablation_macd_diff_dif_dea_entries_1_20250205_27"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
#     # 生成图表并保存为HTML
#     stock_portfolio = portfolio[symbol]
#     fig = stock_portfolio.plot()  # 生成图表
#     fig.write_html(f"{path}/seagull/html/{symbol}_portfolio_plot.html")  # 保存为HTML文件
# =============================================================================
# =============================================================================
# 2025-02-07 10:41:19.463 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 23158 2022-08-04 2024-02-02     367      10000.0   8665.358       -13.346   
# 23159 2022-08-04 2024-02-02     367      10000.0   9454.335        -5.457   
# 23160 2022-08-04 2024-02-02     367      10000.0   8870.111       -11.299   
# 23161 2022-08-04 2024-02-02     367      10000.0   9744.871        -2.551   
# 23162 2022-08-04 2024-02-02     367      10000.0   8181.583       -18.184   
#          ...        ...     ...          ...        ...           ...   
# 23736 2022-08-04 2024-02-02     367      10000.0  10208.874         2.089   
# 23737 2022-08-04 2024-02-02     367      10000.0   8197.765       -18.022   
# 23738 2022-08-04 2024-02-02     367      10000.0   9385.828        -6.142   
# 23739 2022-08-04 2024-02-02     367      10000.0   8329.206       -16.708   
# 23740 2022-08-04 2024-02-02     367      10000.0   8087.181       -19.128   
# 
#        benchmark_return  ann_return  max_dd  calmar_ratio  bar_num  \
# 23158           -13.803      -9.049  14.136        -0.640      367   
# 23159             6.989      -3.647  14.867        -0.245      367   
# 23160           -16.031      -7.632  12.044        -0.634      367   
# 23161             5.951      -1.697  11.864        -0.143      367   
# 23162           -29.205     -12.444  19.156        -0.650      367   
#                 ...         ...     ...           ...      ...   
# 23736           -22.427       1.378  32.287         0.043      367   
# 23737           -41.660     -12.329  31.100        -0.396      367   
# 23738            -7.714      -4.110  10.087        -0.407      367   
# 23739           -29.471     -11.401  30.073        -0.379      367   
# 23740           -23.096     -13.114  40.696        -0.322      367   
# 
#        price_start  price_end  total_open_trades  total_closed_trades  
# 23158        2.927      2.523                  1                   22  
# 23159        0.744      0.796                  1                   24  
# 23160        2.701      2.268                  1                   20  
# 23161        1.882      1.994                  1                   23  
# 23162        0.654      0.463                  0                   23  
#            ...        ...                ...                  ...  
# 23736        0.758      0.588                  0                   19  
# 23737        1.217      0.710                  0                   23  
# 23738        1.076      0.993                  0                   24  
# 23739        1.001      0.706                  0                   28  
# 23740        0.788      0.606                  1                   22  
# 
# [583 rows x 15 columns]
# 2025-02-07 10:41:19.473 | INFO     | backtest.analyze:pipeline:165 - base_ann_return_mean: -16.608
# 2025-02-07 10:41:19.475 | INFO     | backtest.analyze:pipeline:166 - strategy_ann_return_mean: -11.758
# 2025-02-07 10:41:19.476 | INFO     | backtest.analyze:pipeline:167 - base_max_dd_mean: 33.459
# 2025-02-07 10:41:19.478 | INFO     | backtest.analyze:pipeline:168 - strategy_max_dd_mean: 26.337
# 2025-02-07 10:41:19.479 | INFO     | backtest.analyze:pipeline:169 - strategy_total_open_trades_mean: 0.396
# 2025-02-07 10:41:19.480 | INFO     | backtest.analyze:pipeline:170 - strategy_total_closed_trades_mean: 24.635
# 2025-02-07 10:41:19.482 | INFO     | backtest.analyze:pipeline:175 - score_mean_base: 8.958
# 2025-02-07 10:41:19.483 | INFO     | backtest.analyze:pipeline:176 - score_mean_strategy_effective: 14.482
# 2025-02-07 10:41:19.486 | INFO     | backtest.analyze:pipeline:187 - strategy_rank: 
# window
# None-None-None    10.799
# dtype: float64
# 2025-02-07 10:41:19.554 | INFO     | backtest.analyze:pipeline:191 - rank_portfolio: 
# full_code
# SH.513100    39.657
# SH.513300    39.387
# SH.513080    38.984
# SZ.159941    38.680
# SZ.159632    38.064
#  
# SH.561910   -14.917
# SZ.159752   -14.941
# SH.562880   -15.048
# SH.561160   -15.623
# SZ.159796   -15.742
# Length: 583, dtype: float64
# 2025-02-07 10:41:19.557 | INFO     | backtest.analyze:pipeline:200 - baseline: 13.439
# 2025-02-07 10:41:19.559 | INFO     | backtest.analyze:pipeline:201 - baseline_strategy: 17.633
# 2025-02-07 10:41:19.692 | INFO     | backtest.analyze:pipeline:207 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.516350  -1.638000  24.478000        26.116000
# SH.513120   0.259000  22.100000        21.841000
# SZ.159857 -15.675000   5.660000        21.335000
# SZ.159859  -6.058000  15.139000        21.197000
# SH.516290 -15.507000   5.588000        21.095000
#              ...        ...              ...
# SZ.159003  29.893000  12.861000       -17.032000
# SH.511900  29.890000  12.382000       -17.508000
# SZ.159001  29.896000  12.165000       -17.731000
# SH.513800  40.352000  20.942000       -19.410000
# mean        8.957539  14.481616         5.524077
# 
# [584 rows x 3 columns]
# 2025-02-07 10:41:19.699 | INFO     | backtest.analyze:pipeline:211 - rank_personal: 
#       full_code comparison_experiment          window  bar_num  ann_return  \
# 4039  SH.513300                  base  None-None-None      367      26.203   
# 541   SH.513300                  base  None-None-None      367      26.203   
# 4029  SH.513100                  base  None-None-None      367      25.282   
# 531   SH.513100                  base  None-None-None      367      25.282   
# 4473  SZ.159941                  base  None-None-None      367      24.773   
#         ...                   ...             ...      ...         ...   
# 4266  SH.562880                  base  None-None-None      367     -44.404   
# 161   SH.561160                  base  None-None-None      367     -44.924   
# 4242  SH.561160                  base  None-None-None      367     -44.924   
# 299   SZ.159796                  base  None-None-None      367     -45.011   
# 4380  SZ.159796                  base  None-None-None      367     -45.011   
# 
#       max_dd   score  
# 4039  14.739  43.638  
# 541   14.739  43.638  
# 4029  16.136  42.645  
# 531   16.136  42.645  
# 4473  16.667  42.112  
#      ...     ...  
# 4266  61.804 -17.890  
# 161   62.090 -18.336  
# 4242  62.090 -18.336  
# 299   62.324 -18.457  
# 4380  62.324 -18.457  
# 
# [1749 rows x 7 columns]
# =============================================================================
