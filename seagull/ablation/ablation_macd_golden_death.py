# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
(ablation_macd_golden_death)
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
        
        #金叉：Golden Cross
        #死叉：Death Cross
        # 买入信号：MACD 线从下方突破信号线（金叉）
        entries = macd.macd > macd.signal  # DIF > DEA (金叉)
        
        # 卖出信号：MACD 线从上方跌破信号线（死叉）
        exits = macd.macd < macd.signal  # DIF < DEA (死叉)
        
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
                                                   'init_cash': 10000,
                                                   },
                                 strategy_params_list=strategy_params_list,
                                 )
    comparison_experiment = "ablation_macd_diff_dif_dea_entries_1_20250205_7"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2025-02-05 22:29:45.507 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 14388 2022-08-04 2024-02-02     367      10000.0   8710.244       -12.898   
# 14389 2022-08-04 2024-02-02     367      10000.0  10403.341         4.033   
# 14390 2022-08-04 2024-02-02     367      10000.0   8821.038       -11.790   
# 14391 2022-08-04 2024-02-02     367      10000.0  10185.602         1.856   
# 14392 2022-08-04 2024-02-02     367      10000.0   6698.342       -33.017   
#          ...        ...     ...          ...        ...           ...   
# 14966 2022-08-04 2024-02-02     367      10000.0  10332.513         3.325   
# 14967 2022-08-04 2024-02-02     367      10000.0   8461.294       -15.387   
# 14968 2022-08-04 2024-02-02     367      10000.0   7986.326       -20.137   
# 14969 2022-08-04 2024-02-02     367      10000.0   9289.948        -7.101   
# 14970 2022-08-04 2024-02-02     367      10000.0   8306.952       -16.930   
# 
#        benchmark_return  max_dd  bar_num  price_start  price_end  \
# 14388           -13.803  17.007      367        2.927      2.523   
# 14389             6.989  12.882      367        0.744      0.796   
# 14390           -16.031  21.400      367        2.701      2.268   
# 14391             5.951   9.364      367        1.882      1.994   
# 14392           -29.205  33.017      367        0.654      0.463   
#                 ...     ...      ...          ...        ...   
# 14966           -22.427  27.835      367        0.758      0.588   
# 14967           -41.660  32.796      367        1.217      0.710   
# 14968            -7.714  20.137      367        1.076      0.993   
# 14969           -29.471  28.676      367        1.001      0.706   
# 14970           -23.096  43.658      367        0.788      0.606   
# 
#        total_open_trades  total_closed_trades  
# 14388                  1                   11  
# 14389                  1                   11  
# 14390                  1                   11  
# 14391                  1                   10  
# 14392                  0                   14  
#                  ...                  ...  
# 14966                  0                   10  
# 14967                  0                   14  
# 14968                  0                   12  
# 14969                  0                   13  
# 14970                  1                   10  
# 
# [583 rows x 13 columns]
# 2025-02-05 22:29:45.520 | INFO     | backtest.analyze:pipeline:158 - strategy_total_open_trades_mean: 0.235
# 2025-02-05 22:29:45.528 | INFO     | backtest.analyze:pipeline:159 - strategy_total_closed_trades_mean: 11.768
# 2025-02-05 22:29:45.537 | INFO     | backtest.analyze:pipeline:169 - score_mean_base: 8.958
# 2025-02-05 22:29:45.544 | INFO     | backtest.analyze:pipeline:170 - score_mean_strategy_effective: 16.206
# 2025-02-05 22:29:45.554 | INFO     | backtest.analyze:pipeline:181 - strategy_rank: 
# window
# 12-26-9           16.206
# None-None-None     8.958
# dtype: float64
# 2025-02-05 22:29:45.623 | INFO     | backtest.analyze:pipeline:185 - rank_portfolio: 
# full_code
# SH.513100    39.536
# SH.513030    38.446
# SZ.159941    38.070
# SH.513080    37.451
# SH.517180    37.218
#  
# SH.562880   -12.252
# SH.561160   -12.360
# SZ.159796   -12.916
# SH.516270   -13.009
# SZ.159752   -13.425
# Length: 583, dtype: float64
# 2025-02-05 22:29:45.631 | INFO     | backtest.analyze:pipeline:194 - baseline: 13.439
# 2025-02-05 22:29:45.638 | INFO     | backtest.analyze:pipeline:195 - baseline_strategy: 15.115
# 2025-02-05 22:29:45.767 | INFO     | backtest.analyze:pipeline:201 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.512480  -3.055000  33.405000        36.460000
# SH.516920  -2.460000  29.846000        32.306000
# SH.516640  -2.348000  28.578000        30.926000
# SH.516350  -1.638000  29.132000        30.770000
# SH.513380   4.980000  34.586000        29.606000
#              ...        ...              ...
# SH.512890  34.422000  21.501000       -12.921000
# SH.513300  43.638000  30.054000       -13.584000
# SH.516770  22.143000   8.444000       -13.699000
# SH.513360  35.194000  15.367000       -19.827000
# mean        8.957539  16.205835         7.248297
# 
# [584 rows x 3 columns]
# 2025-02-05 22:29:45.779 | INFO     | backtest.analyze:pipeline:205 - rank_personal: 
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
# [1166 rows x 7 columns]
# =============================================================================
