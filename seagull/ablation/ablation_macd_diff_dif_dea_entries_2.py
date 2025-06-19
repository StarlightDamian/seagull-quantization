# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_dif_dea_entries_2
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
            macd_ewm=False, # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
            signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
            adjust=False, #布尔值，是否在计算EMA时进行调整
            #cache_dict,字典，用于缓存计算结果
        )
        
        # 计算 DIF 和 DEA 的斜率
        dif_slope = macd.macd.diff()  # DIF 线的斜率
        dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 交易信号
        #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
        #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        entries = (dif_slope > 0) & (dif_slope.shift(1) <= 0)# & (dea_slope > 0) & (dea_slope.shift(1) < 0)
        
        # 卖出信号：DIF 的斜率从正变负
        exits = (dif_slope <= 0) & (dif_slope.shift(1) > 0)
        
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
    comparison_experiment = "ablation_macd_diff_dif_dea_entries_0_20250204"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2024-10-28 14:16:16.306 | INFO     | backtest.analyze:pipeline:154 -                      start                  end  period  start_value  \
# 8361   2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 9324   2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 9586   2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 10311  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 10312  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
#                    ...                  ...     ...          ...   
# 76429  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 76430  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 76432  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 76433  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 76434  2019-01-02 00:00:00  2022-12-30 00:00:00     972      10000.0   
# 
#        end_value  total_return  benchmark_return  max_gross_exposure  \
# 8361    9846.518        -1.535            -7.813               100.0   
# 9324   10222.180         2.222           -42.002               100.0   
# 9586    9835.553        -1.644            -7.813               100.0   
# 10311   9120.834        -8.792            44.979               100.0   
# 10312   8905.323       -10.947            23.225               100.0   
#          ...           ...               ...                 ...   
# 76429   9225.764        -7.742             0.754               100.0   
# 76430   9790.446        -2.096           -36.328               100.0   
# 76432   8795.441       -12.046            10.549               100.0   
# 76433   9119.582        -8.804           -10.521               100.0   
# 76434  10952.142         9.521           -16.038               100.0   
# 
#        total_fees_paid  max_dd  ... slippage  freq_portfolio  bar_num  \
# 8361             9.990   4.075  ...    0.001               d      201   
# 9324             9.990   0.337  ...    0.001               d      363   
# 9586             9.990   4.075  ...    0.001               d      201   
# 10311          191.083  14.061  ...    0.001               d      969   
# 10312          133.358  11.561  ...    0.001               d      968   
#                ...     ...  ...      ...             ...      ...   
# 76429           57.810   7.742  ...    0.001               d      708   
# 76430           39.450   2.604  ...    0.001               d      693   
# 76432          166.464  16.272  ...    0.001               d      682   
# 76433           39.412  13.156  ...    0.001               d      678   
# 76434           59.625   5.309  ...    0.001               d      663   
# 
#        date_start   date_end  price_start  price_end  \
# 8361   2022-03-08 2022-12-30        0.960      0.885   
# 9324   2021-07-07 2022-12-30        1.019      0.591   
# 9586   2022-03-08 2022-12-30        0.960      0.885   
# 10311  2019-01-02 2022-12-30        0.936      1.357   
# 10312  2019-01-02 2022-12-30        2.338      2.881   
#           ...        ...          ...        ...   
# 76429  2020-02-07 2022-12-30        0.928      0.935   
# 76430  2020-02-28 2022-12-30        1.024      0.652   
# 76432  2020-03-16 2022-12-30        0.910      1.006   
# 76433  2020-03-20 2022-12-30        0.960      0.859   
# 76434  2020-04-13 2022-12-30        0.954      0.801   
# 
#                             primary_key     insert_timestamp    window  
# 8361   949fda97dcda5502c32ce5f1d2f4ead5  2024-10-24 19:25:48   10-26-9  
# 9324   a6967275a8107eb86b20c67d97e2231e  2024-10-24 19:25:48   11-27-7  
# 9586   19e058749336af0c0ca383e2964f8f64  2024-10-24 19:25:48  12-26-10  
# 10311  4f77d592b5720d50342c281194e39eff  2024-10-24 19:25:48  10-24-10  
# 10312  2509a2166d7b10d9a52e0fc4c579d6a0  2024-10-24 19:25:48  10-24-10  
#                                 ...                  ...       ...  
# 76429  f89000056858c98d33c8877514bf1621  2024-10-24 19:25:48  14-28-10  
# 76430  c5f680c1d1a237838be88038577a5cec  2024-10-24 19:25:48  14-28-10  
# 76432  21eb0bc8ea78fc6c819b3b205b1eadce  2024-10-24 19:25:48  14-28-10  
# 76433  bc76cf80657c5b6ab8ef8a4e79d0bb57  2024-10-24 19:25:48  14-28-10  
# 76434  5f496827ef8f7512e38fa7e8588d6485  2024-10-24 19:25:48  14-28-10  
# 
# [12688 rows x 56 columns]
# 2024-10-28 14:16:16.327 | INFO     | backtest.analyze:pipeline:164 - score_mean_base: 20.657
# 2024-10-28 14:16:16.328 | INFO     | backtest.analyze:pipeline:165 - score_mean_strategy_effective: 27.767
# 2024-10-28 14:16:16.342 | INFO     | backtest.analyze:pipeline:176 - strategy_rank: 
# window
# None-None-None    30.780
# 11-27-7           30.239
# 10-26-9           28.628
# 10-24-10          28.384
# 10-28-10          28.319
#  
# 11-27-10          27.660
# 14-28-10          27.651
# 13-27-10          27.620
# 11-25-10          27.584
# 14-27-10          27.270
# Length: 28, dtype: float64
# 2024-10-28 14:16:16.479 | INFO     | backtest.analyze:pipeline:180 - rank_portfolio: 
# full_code
# 516860       49.705
# 159851       48.397
# 516100       47.915
# 513090       46.995
# 159923       41.715
#  
# SZ.159667    24.191
# SH.513260    23.261
# SH.510680    23.095
# SZ.159636    20.098
# SH.513360    15.370
# Length: 1599, dtype: float64
# 2024-10-28 14:16:16.483 | INFO     | backtest.analyze:pipeline:189 - baseline: 26.084
# 2024-10-28 14:16:16.484 | INFO     | backtest.analyze:pipeline:190 - baseline_strategy: 29.67
# 2024-10-28 14:16:16.599 | INFO     | backtest.analyze:pipeline:196 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.513330  -0.484000  26.570000        27.054000
# SH.513360   2.787000  27.952000        25.165000
# SH.513130   3.025000  27.867000        24.842000
# SH.513980   3.602000  28.135000        24.533000
# SH.513860   3.781000  27.965000        24.184000
#              ...        ...              ...
# SH.512600  35.974000  27.789000        -8.185000
# SZ.159949  35.275000  26.861000        -8.414000
# SH.512520  37.808000  28.233000        -9.575000
# SH.512690  62.992000  25.744000       -37.248000
# mean       20.402215  28.073598         7.671379
# 
# [663 rows x 3 columns]
# 2024-10-28 14:16:16.617 | INFO     | backtest.analyze:pipeline:200 - rank_personal: 
#       full_code comparison_experiment          window  bar_num  ann_return  \
# 698      516860                  base  None-None-None      248      66.124   
# 284      159851                  base  None-None-None      248      62.275   
# 644      516100                  base  None-None-None      248      61.644   
# 515      513090                  base  None-None-None      248      55.740   
# 5576  SH.512690                  base  None-None-None      892      68.997   
#         ...                   ...             ...      ...         ...   
# 5978  SZ.159740                  base  None-None-None      391     -14.014   
# 5609  SH.513130                  base  None-None-None      388     -14.447   
# 5983  SZ.159747                  base  None-None-None      358     -14.863   
# 5623  SH.513360                  base  None-None-None      377     -12.543   
# 5622  SH.513330                  base  None-None-None      460     -17.573   
# 
#       max_dd   score  
# 698   10.327  72.546  
# 284   10.293  69.878  
# 644   11.364  69.129  
# 515    6.393  66.409  
# 5576  49.451  62.992  
#      ...     ...  
# 5978  62.231   3.370  
# 5609  62.330   3.025  
# 5983  61.561   2.941  
# 5623  68.663   2.787  
# 5622  67.446  -0.484  
# 
# [17020 rows x 7 columns]
# 
# =============================================================================
