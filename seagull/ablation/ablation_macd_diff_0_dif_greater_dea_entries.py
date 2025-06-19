# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_0_dif_greater_dea_entries
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd

from __init__ import path
from utils import utils_log
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
        dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 交易信号
        #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
        #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        entries = (dif_slope > dea_slope) & (dif_slope.shift(1) < dea_slope.shift(1)) 
        
        # 卖出信号：DIF 的斜率从正变负
        exits = (dif_slope < 0) & (dif_slope.shift(1) > 0)
        
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
    comparison_experiment = "ablation_macd_diff_0_dif_greater_dea_entries_20241011_2"
    ablation_macd.ablation_experiment(date_start='2019-01-01', 
                                               date_end='2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2024-10-11 21:25:00.000 | SUCCESS  | utils.utils_data:output_database:34 - Writing to database conclusion-succeeded.
# 2024-10-11 21:25:03.246 | INFO     | backtest.analyze:pipeline:163 - score_mean_base: 20.657
# 2024-10-11 21:25:03.246 | INFO     | backtest.analyze:pipeline:164 - score_mean_strategy_effective: 18.297
# 2024-10-11 21:25:03.265 | INFO     | backtest.analyze:pipeline:175 - strategy_rank: 
# window
# 13-27-7     29.568
# 12-24-9     29.560
# 14-28-7     29.427
# 13-28-7     29.427
# 12-26-7     29.093
#  
# 12-24-10    17.183
# 14-25-10    16.886
# 13-25-10    16.874
# 14-24-10    16.827
# 13-24-10    16.761
# Length: 38, dtype: float64
# 2024-10-11 21:25:03.330 | INFO     | backtest.analyze:pipeline:179 - rank_portfolio: 
# full_code
# SH.515080    33.246
# SH.510170    33.141
# SH.510880    31.590
# SH.515180    31.388
# SH.510410    31.255
#  
# SH.513050     4.030
# SH.513580     3.629
# SH.513180     3.196
# SH.513010     3.157
# SH.513330     2.949
# Length: 688, dtype: float64
# 2024-10-11 21:25:03.334 | INFO     | backtest.analyze:pipeline:187 - baseline: 26.084
# 2024-10-11 21:25:03.334 | INFO     | backtest.analyze:pipeline:188 - baseline_strategy: 22.197
# 2024-10-11 21:25:03.450 | INFO     | backtest.analyze:pipeline:192 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.513360   2.787000  24.942000        22.155000
# SH.560800  14.029000  24.436000        10.407000
# SZ.159738  18.252000  28.436000        10.184000
# SZ.159859   6.812000  16.912000        10.100000
# SH.516350  11.194000  21.294000        10.100000
#              ...        ...              ...
# SH.511700  29.931000   7.661000       -22.270000
# SZ.159949  35.275000  12.587000       -22.688000
# SZ.159956  32.665000   4.625000       -28.040000
# SH.512690  62.992000  27.315000       -35.677000
# mean       20.515503  18.487862        -2.027641
# 
# [675 rows x 3 columns]
# 2024-10-11 21:25:03.466 | INFO     | backtest.analyze:pipeline:196 - rank_personal: 
#        full_code                              comparison_experiment  \
# 110    SH.512690                                               base   
# 71646  SH.512690  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 57985  SH.515220  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 30312  SH.512040  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 30422  SH.515080  ablation_macd_diff_0_dif_greater_dea_entries_2...   
#          ...                                                ...   
# 19683  SZ.159740  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 72101  SZ.159822  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 50196  SZ.159956  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 22057  SH.513010  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 61187  SZ.159939  ablation_macd_diff_0_dif_greater_dea_entries_2...   
# 
#                window  bar_num  ann_return  max_dd   score  
# 110    None-None-None      892      68.997  49.451  62.992  
# 71646        14-28-10      892      41.000  43.734  45.803  
# 57985        13-28-10      692      27.537  17.382  43.585  
# 30312        11-28-10      972      22.294  13.262  41.463  
# 30422        11-28-10      731      20.310   9.186  41.291  
#               ...      ...         ...     ...     ...  
# 19683        11-24-10      391     -23.885  66.436  -4.320  
# 72101        14-28-10      535     -23.827  67.499  -4.436  
# 50196        13-25-10      971     -23.870  69.180  -4.904  
# 22057        11-25-10      393     -26.277  70.613  -6.943  
# 61187        14-24-10      972     -26.290  71.277  -7.261  
# 
# [17438 rows x 7 columns]
# =============================================================================
