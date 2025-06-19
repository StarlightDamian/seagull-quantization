# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_dif_dea_entries_0

ads_info_incr_bacetest
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_log
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class ablationMacd(vectorbt_macd.backtestVectorbtMacd, analyze.backtestAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macd_hist = None
    
# =============================================================================
#     def strategy(self, subtable_df, data):
#         window_fast, window_slow, window_signal = subtable_df[['window_fast', 'window_slow', 'window_signal']].values[0]
# 
#         # https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/indicators/nb.py#L171
#         macd = vbt.MACD.run(
#             close=data,  # close: 2D数组，表示收盘价
#             fast_window=window_fast,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
#             slow_window=window_slow,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
#             signal_window=window_signal,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
#             macd_ewm=False, # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
#             signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
#             adjust=False, #布尔值，是否在计算EMA时进行调整
#             #cache_dict,字典，用于缓存计算结果
#         )
#         
#         # 计算 DIF 和 DEA 的斜率
#         dif_slope = macd.macd.diff()  # DIF 线的斜率
#         dea_slope = macd.signal.diff()  # DEA 线的斜率
#         
#         # 交易信号
#         #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
#         #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
#         # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
#         entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0) & (dea_slope.shift(1) < 0)
#         
#         # 卖出信号：DIF 的斜率从正变负
#         exits = (dif_slope < 0) & (dif_slope.shift(1) > 0)
#         
#         entries_exits_t = pd.concat([entries, exits], axis=1, keys=['entries', 'exits']).T
#         return entries_exits_t
# =============================================================================
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
        entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0) & (dea_slope.shift(1) < 0)
        
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
    comparison_experiment = "ablation_macd_diff_dif_dea_entries_0_20250204_26"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    #strategy_df.ann_return
    
# =============================================================================
# 2025-02-05 21:56:10.908 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 11104 2022-08-04 2024-02-02     367      10000.0   9880.399        -1.196   
# 11105 2022-08-04 2024-02-02     367      10000.0   9642.901        -3.571   
# 11106 2022-08-04 2024-02-02     367      10000.0   9866.558        -1.334   
# 11108 2022-08-04 2024-02-02     367      10000.0  10738.535         7.385   
# 11109 2022-08-04 2024-02-02     367      10000.0   9256.207        -7.438   
#          ...        ...     ...          ...        ...           ...   
# 11838 2022-08-04 2024-02-02     367      10000.0  10119.187         1.192   
# 11839 2022-08-04 2024-02-02     367      10000.0   9614.929        -3.851   
# 11840 2022-08-04 2024-02-02     367      10000.0   9783.124        -2.169   
# 11841 2022-08-04 2024-02-02     367      10000.0  10023.052         0.231   
# 11842 2022-08-04 2024-02-02     367      10000.0   9725.161        -2.748   
# 
#        benchmark_return  max_dd  bar_num  price_start  price_end  \
# 11104             2.703   1.196      367        0.962      0.988   
# 11105           -13.803   3.571      367        2.927      2.523   
# 11106             6.989   1.840      367        0.744      0.796   
# 11108             5.951   1.964      367        1.882      1.994   
# 11109           -29.205   7.857      367        0.654      0.463   
#                 ...     ...      ...          ...        ...   
# 11838           -49.790   2.326      367        0.954      0.479   
# 11839           -25.235   3.851      367        0.852      0.637   
# 11840           -35.155   2.169      367        0.933      0.605   
# 11841           -19.522   0.200      367        0.963      0.775   
# 11842           -25.310   2.779      367        0.968      0.723   
# 
#        total_open_trades  total_closed_trades  
# 11104                  0                    1  
# 11105                  0                    1  
# 11106                  0                    1  
# 11108                  0                    2  
# 11109                  0                    2  
#                  ...                  ...  
# 11838                  0                    1  
# 11839                  0                    1  
# 11840                  0                    1  
# 11841                  0                    1  
# 11842                  0                    3  
# 
# [410 rows x 13 columns]
# 2025-02-05 21:56:10.919 | INFO     | backtest.analyze:pipeline:158 - strategy_total_open_trades_mean: 0.061
# 2025-02-05 21:56:10.924 | INFO     | backtest.analyze:pipeline:159 - strategy_total_closed_trades_mean: 1.846
# 2025-02-05 21:56:10.930 | INFO     | backtest.analyze:pipeline:169 - score_mean_base: 8.958
# 2025-02-05 21:56:10.934 | INFO     | backtest.analyze:pipeline:170 - score_mean_strategy_effective: 28.487
# 2025-02-05 21:56:10.940 | INFO     | backtest.analyze:pipeline:181 - strategy_rank: 
# window
# 12-26-9           28.487
# None-None-None     8.958
# dtype: float64
# 2025-02-05 21:56:11.004 | INFO     | backtest.analyze:pipeline:185 - rank_portfolio: 
# full_code
# SH.513100    42.645
# SZ.159941    42.112
# SZ.159632    41.328
# SH.517180    40.480
# SH.515450    39.640
#  
# SZ.159752   -16.981
# SZ.159757   -17.001
# SZ.159767   -17.031
# SZ.159840   -17.697
# SZ.159755   -17.835
# Length: 583, dtype: float64
# 2025-02-05 21:56:11.010 | INFO     | backtest.analyze:pipeline:194 - baseline: 13.439
# 2025-02-05 21:56:11.015 | INFO     | backtest.analyze:pipeline:195 - baseline_strategy: 29.903
# 2025-02-05 21:56:11.119 | INFO     | backtest.analyze:pipeline:201 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SZ.159775 -17.761000  30.223000        47.984000
# SH.516390 -15.541000  29.444000        44.985000
# SH.516160 -16.653000  27.894000        44.547000
# SH.516850 -16.674000  27.647000        44.321000
# SH.516580 -16.507000  27.168000        43.675000
#              ...        ...              ...
# SZ.159866  41.677000  30.197000       -11.480000
# SH.513080  40.620000  28.441000       -12.179000
# SH.513030  41.691000  27.838000       -13.853000
# SH.513300  43.638000  28.098000       -15.540000
# mean        8.674854  28.487393        19.812539
# 
# [411 rows x 3 columns]
# 2025-02-05 21:56:11.129 | INFO     | backtest.analyze:pipeline:205 - rank_personal: 
#      full_code comparison_experiment          window  bar_num  ann_return  \
# 120  SH.513300                  base  None-None-None      367      26.203   
# 110  SH.513100                  base  None-None-None      367      25.282   
# 554  SZ.159941                  base  None-None-None      367      24.773   
# 104  SH.513030                  base  None-None-None      367      22.893   
# 507  SZ.159866                  base  None-None-None      367      23.151   
# ..         ...                   ...             ...      ...         ...   
# 445  SZ.159775                  base  None-None-None      367     -44.745   
# 434  SZ.159755                  base  None-None-None      367     -44.734   
# 347  SH.562880                  base  None-None-None      367     -44.404   
# 323  SH.561160                  base  None-None-None      367     -44.924   
# 461  SZ.159796                  base  None-None-None      367     -45.011   
# 
#      max_dd   score  
# 120  14.739  43.638  
# 110  16.136  42.645  
# 554  16.667  42.112  
# 104  13.654  41.691  
# 507  13.793  41.677  
# ..      ...     ...  
# 445  60.600 -17.761  
# 434  60.621 -17.835  
# 347  61.804 -17.890  
# 323  62.090 -18.336  
# 461  62.324 -18.457  
# 
# [993 rows x 7 columns]
# =============================================================================
