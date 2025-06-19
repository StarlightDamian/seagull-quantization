# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_dif_0
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
        
        # 计算 DIF 和 DEA 的斜率
        dif_slope = macd.macd.diff()  # DIF 线的斜率
        dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 交易信号
        #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
        #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) #& (dea_slope > 0) & (dea_slope.shift(1) < 0)
        
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
    comparison_experiment = "ablation_macd_diff_dif_0_20241011"
    ablation_macd.ablation_experiment(date_start='2019-01-01', 
                                               date_end='2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2024-10-11 21:01:46.621 | INFO     | backtest.analyze:pipeline:163 - score_mean_base: 20.657
# 2024-10-11 21:01:46.623 | INFO     | backtest.analyze:pipeline:164 - score_mean_strategy_effective: 17.605
# 2024-10-11 21:01:46.637 | INFO     | backtest.analyze:pipeline:175 - strategy_rank: 
# window
# None-None-None    20.657
# 10-28-10          19.924
# 12-28-10          19.824
# 10-27-10          19.783
# 12-27-10          19.755
#  
# 13-25-10          17.265
# 13-24-10          17.173
# 14-24-10          17.099
# 10-24-7           17.073
# 14-25-10          16.699
# Length: 27, dtype: float64
# 2024-10-11 21:01:46.704 | INFO     | backtest.analyze:pipeline:179 - rank_portfolio: 
# full_code
# SH.512690    34.924
# SH.510170    30.277
# SH.513120    29.862
# SZ.159776    29.397
# SH.588350    29.319
#  
# SZ.159998     6.063
# SH.511620     5.186
# SH.513330     5.038
# SH.512720     4.944
# SH.513050     4.773
# Length: 688, dtype: float64
# 2024-10-11 21:01:46.708 | INFO     | backtest.analyze:pipeline:187 - baseline: 26.084
# 2024-10-11 21:01:46.709 | INFO     | backtest.analyze:pipeline:188 - baseline_strategy: 18.202
# 2024-10-11 21:01:46.829 | INFO     | backtest.analyze:pipeline:192 - comparison_portfolio: 
#                 base  strategy  strategy_better
# full_code                                      
# SH.513360   2.787000  18.52200        15.735000
# SH.516820   5.044000  17.47400        12.430000
# SZ.159786  10.251000  22.17500        11.924000
# SH.562950  13.146000  24.66600        11.520000
# SH.516350  11.194000  22.48300        11.289000
#              ...       ...              ...
# SH.511600  29.678000   5.77000       -23.908000
# SH.511860  29.680000   5.52200       -24.158000
# SH.511700  29.931000   5.48500       -24.446000
# SH.512690  62.992000  33.80100       -29.191000
# mean       20.522047  18.67613        -1.845917
# 
# [676 rows x 3 columns]
# 2024-10-11 21:01:46.849 | INFO     | backtest.analyze:pipeline:196 - rank_personal: 
#        full_code        comparison_experiment          window  bar_num  \
# 110    SH.512690                         base  None-None-None      892   
# 66316  SH.512690  macd_diff_20241011_diff_0_5        14-27-10      892   
# 63561  SH.512690  macd_diff_20241011_diff_0_5        14-26-10      892   
# 11027  SH.512690  macd_diff_20241011_diff_0_5        10-27-10      892   
# 52366  SH.512690  macd_diff_20241011_diff_0_5        13-27-10      892   
#          ...                          ...             ...      ...   
# 11006  SH.512290  macd_diff_20241011_diff_0_5        10-27-10      883   
# 5081   SH.512120  macd_diff_20241011_diff_0_5        10-25-10      971   
# 63564  SH.512720  macd_diff_20241011_diff_0_5        14-26-10      820   
# 60636  SH.512720  macd_diff_20241011_diff_0_5        14-25-10      820   
# 57879  SH.512720  macd_diff_20241011_diff_0_5        14-24-10      820   
# 
#        ann_return  max_dd   score  
# 110        68.997  49.451  62.992  
# 66316      48.827  34.142  53.821  
# 63561      39.471  37.597  46.391  
# 11027      38.459  36.725  45.809  
# 52366      36.980  36.043  45.052  
#           ...     ...     ...  
# 11006     -18.625  67.918  -0.960  
# 5081      -21.039  64.747  -1.877  
# 63564     -20.948  68.120  -2.729  
# 60636     -22.240  67.689  -3.529  
# 57879     -22.645  68.836  -4.081  
# 
# [17511 rows x 7 columns]
# =============================================================================
