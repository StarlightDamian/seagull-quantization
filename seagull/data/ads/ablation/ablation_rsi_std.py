# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:13:55 2025

@author: awei

ablation_rsi_std
"""
import os
import itertools

import numpy as np
import vectorbt as vbt
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_log, utils_character
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

# =============================================================================
# def winsorize(df, window=50):
#     """
#     极值处理，逐列进行裁剪（使用向量化操作）。
#     """
#     return df.rolling(window=window).std()
# =============================================================================

def winsorize(df, window=50, n_lower=3, n_upper=3):
    """
    极值处理，逐列进行裁剪（使用向量化操作）。
    lower.null = std_window+rsi_window-1 = 63
    rsi.null=rsi_window = 14
    """
    # 计算每列的均值和标准差
    means = df.rolling(window=window).mean()#axis=0
    stds = df.rolling(window=window).std()#axis=0

    # 计算上限和下限
    upper = means + n_upper * stds
    lower = means - n_lower * stds
    return lower, upper

    
class ablationMacd(vectorbt_macd.backtestVectorbtMacd, analyze.backtestAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macd_hist = None
    
    def strategy(self, subtable_df, data):
        window_fast, window_slow, window_signal = subtable_df[['window_fast', 'window_slow', 'window_signal']].values[0]
        
        #global data1
        #data1 = data
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

        # RSI计算
        rsi = vbt.RSI.run(data, window=14).rsi
        
        global rsi1
        rsi1 = rsi
        # 动态标准差计算
        #lower, upper = winsorize(rsi, window=50, n_lower=3.5, n_upper=2)  # 50天的RSI标准差
        lower, upper = winsorize(rsi, window=40, n_lower=3.5, n_upper=2)
        
        # 买入信号：当前RSI小于窗口内的标准差
        entries = rsi < lower  # RSI小于标准差，视为买入信号

        # 卖出信号：当前RSI大于窗口内的标准差
        exits = rsi > upper  # RSI大于标准差，视为卖出信号
        
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
    comparison_experiment = utils_character.generate_random_string(6)  # "ablation_macd_diff_dif_dea_entries_1_20250205_27"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    strategy_df = strategy_df.sort_values(by='score', ascending=False)

    strategy_df.benchmark_return.mean()
    #Out[93]: -2.1680833333333336

    base_df.benchmark_return.mean()
    #Out[94]: -22.97705488850772
    
    # output
    eval_columns = ['full_code','ann_return','max_dd','calmar_ratio',
                    'total_closed_trades','win_rate','profit_loss_ratio']
    base_df[eval_columns].to_csv(f'{PATH}/data/eval_base.csv', index=False)
    strategy_df[eval_columns].to_csv(f'{PATH}/data/eval_strategy.csv', index=False)
    
# =============================================================================
# 2025-02-07 14:57:33.179 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 38355 2022-08-04 2024-02-02     367      10000.0   9870.740        -1.293   
# 38356 2022-08-04 2024-02-02     367      10000.0   9966.953        -0.330   
# 38358 2022-08-04 2024-02-02     367      10000.0   9923.891        -0.761   
# 38360 2022-08-04 2024-02-02     367      10000.0   9961.474        -0.385   
# 38361 2022-08-04 2024-02-02     367      10000.0   9961.076        -0.389   
#          ...        ...     ...          ...        ...           ...   
# 38697 2022-08-04 2024-02-02     367      10000.0   9229.443        -7.706   
# 38704 2022-08-04 2024-02-02     367      10000.0  10733.993         7.340   
# 38710 2022-08-04 2024-02-02     367      10000.0   9751.434        -2.486   
# 38782 2022-08-04 2024-02-02     367      10000.0  10437.618         4.376   
# 38879 2022-08-04 2024-02-02     367      10000.0  11937.449        19.374   
# 
#        benchmark_return  ann_return  max_dd  calmar_ratio  bar_num  \
# 38355             0.015      -0.858   1.293        -0.664      367   
# 38356             0.007      -0.219   0.330        -0.662      367   
# 38358             0.011      -0.505   0.761        -0.663      367   
# 38360             0.015      -0.255   0.385        -0.663      367   
# 38361             0.014      -0.258   0.389        -0.663      367   
#                 ...         ...     ...           ...      ...   
# 38697           -21.140      -5.171   8.540        -0.605      367   
# 38704            -3.682       4.802  19.419         0.247      367   
# 38710            -5.757      -1.653  28.906        -0.057      367   
# 38782             7.152       2.877  11.737         0.245      367   
# 38879            -5.313      12.441   3.086         4.032      367   
# 
#        price_start  price_end  total_open_trades  total_closed_trades  
# 38355       99.995    100.010                  0                    4  
# 38356      100.003    100.010                  0                    1  
# 38358      100.004    100.015                  0                    2  
# 38360      100.008    100.023                  0                    1  
# 38361       99.998    100.012                  0                    1  
#            ...        ...                ...                  ...  
# 38697        0.842      0.664                  1                    0  
# 38704        0.869      0.837                  0                    2  
# 38710        0.938      0.884                  0                    1  
# 38782        0.797      0.854                  0                    1  
# 38879        1.261      1.194                  0                    1  
# 
# [24 rows x 15 columns]
# 2025-02-07 14:57:33.189 | INFO     | backtest.analyze:pipeline:165 - base_ann_return_mean: -16.608
# 2025-02-07 14:57:33.191 | INFO     | backtest.analyze:pipeline:166 - strategy_ann_return_mean: 3.232
# 2025-02-07 14:57:33.193 | INFO     | backtest.analyze:pipeline:167 - base_max_dd_mean: 33.459
# 2025-02-07 14:57:33.195 | INFO     | backtest.analyze:pipeline:168 - strategy_max_dd_mean: 5.577
# 2025-02-07 14:57:33.197 | INFO     | backtest.analyze:pipeline:169 - strategy_total_open_trades_mean: 0.125
# 2025-02-07 14:57:33.199 | INFO     | backtest.analyze:pipeline:170 - strategy_total_closed_trades_mean: 1.208
# 2025-02-07 14:57:33.201 | INFO     | backtest.analyze:pipeline:175 - score_mean_base: 8.958
# 2025-02-07 14:57:33.203 | INFO     | backtest.analyze:pipeline:176 - score_mean_strategy_effective: 30.631
# 2025-02-07 14:57:33.207 | INFO     | backtest.analyze:pipeline:187 - strategy_rank: 
# window
# None-None-None    9.395
# dtype: float64
# 2025-02-07 14:57:33.272 | INFO     | backtest.analyze:pipeline:191 - rank_portfolio: 
# full_code
# SH.513300    43.638
# SH.513100    42.645
# SZ.159941    42.112
# SH.513030    41.691
# SZ.159866    41.677
#  
# SZ.159775   -17.761
# SZ.159755   -17.835
# SH.562880   -17.890
# SH.561160   -18.336
# SZ.159796   -18.457
# Length: 583, dtype: float64
# 2025-02-07 14:57:33.275 | INFO     | backtest.analyze:pipeline:200 - baseline: 13.439
# 2025-02-07 14:57:33.278 | INFO     | backtest.analyze:pipeline:201 - baseline_strategy: nan
# 2025-02-07 14:57:33.346 | INFO     | backtest.analyze:pipeline:207 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.516950  18.396000  40.285000        21.889000
# SH.515150  17.245000  36.709000        19.464000
# SH.512960  20.945000  38.165000        17.220000
# SZ.159959  21.461000  37.583000        16.122000
# SH.515900  25.136000  40.227000        15.091000
#              ...        ...              ...
# SH.511990  29.870000  29.437000        -0.433000
# SH.511700  29.876000  29.441000        -0.435000
# SH.511900  29.890000  29.297000        -0.593000
# SH.511880  31.089000  29.918000        -1.171000
# mean       24.463333  30.630917         6.167583
# 
# [25 rows x 3 columns]
# 2025-02-07 14:57:33.353 | INFO     | backtest.analyze:pipeline:211 - rank_personal: 
#       full_code comparison_experiment          window  bar_num  ann_return  \
# 4039  SH.513300                  base  None-None-None      367      26.203   
# 541   SH.513300                  base  None-None-None      367      26.203   
# 4029  SH.513100                  base  None-None-None      367      25.282   
# 531   SH.513100                  base  None-None-None      367      25.282   
# 392   SZ.159941                  base  None-None-None      367      24.773   
#         ...                   ...             ...      ...         ...   
# 185   SH.562880                  base  None-None-None      367     -44.404   
# 4242  SH.561160                  base  None-None-None      367     -44.924   
# 161   SH.561160                  base  None-None-None      367     -44.924   
# 299   SZ.159796                  base  None-None-None      367     -45.011   
# 4380  SZ.159796                  base  None-None-None      367     -45.011   
# 
#       max_dd   score  
# 4039  14.739  43.638  
# 541   14.739  43.638  
# 4029  16.136  42.645  
# 531   16.136  42.645  
# 392   16.667  42.112  
#      ...     ...  
# 185   61.804 -17.890  
# 4242  62.090 -18.336  
# 161   62.090 -18.336  
# 299   62.324 -18.457  
# 4380  62.324 -18.457  
# 
# [1190 rows x 7 columns]
# 
# =============================================================================
