# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_dif_dea_0
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
        entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0) & (dea_slope.shift(1) < 0)
        
        # 卖出信号：DIF 的斜率从正变负
        exits = (dif_slope < 0) & (dif_slope.shift(1) > 0) & (dea_slope < 0) & (dea_slope.shift(1) > 0)
        
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
    comparison_experiment = "ablation_macd_diff_dif_dea_0_20241011"
    ablation_macd.ablation_experiment(date_start='2019-01-01', 
                                               date_end='2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2024-10-11 21:07:33.383 | INFO     | backtest.analyze:pipeline:163 - score_mean_base: 20.657
# 2024-10-11 21:07:33.385 | INFO     | backtest.analyze:pipeline:164 - score_mean_strategy_effective: 24.237
# 2024-10-11 21:07:33.397 | INFO     | backtest.analyze:pipeline:175 - strategy_rank: 
# window
# 12-24-10          24.566
# 10-24-7           24.483
# 11-28-10          24.100
# 10-28-10          23.921
# 12-25-10          23.809
#  
# 10-24-10          23.052
# 11-27-10          22.893
# 13-27-10          22.870
# 11-25-10          22.357
# None-None-None    20.657
# Length: 27, dtype: float64
# 2024-10-11 21:07:33.466 | INFO     | backtest.analyze:pipeline:179 - rank_portfolio: 
# full_code
# SH.560680    31.867
# SH.515220    30.699
# SH.511880    30.270
# SH.512580    30.127
# SZ.159949    29.992
#  
# SH.513060    13.127
# SH.513360    12.847
# SZ.159740    12.571
# SH.513180    12.277
# SZ.159742    11.779
# Length: 688, dtype: float64
# 2024-10-11 21:07:33.470 | INFO     | backtest.analyze:pipeline:187 - baseline: 26.084
# 2024-10-11 21:07:33.472 | INFO     | backtest.analyze:pipeline:188 - baseline_strategy: 28.835
# 2024-10-11 21:07:33.585 | INFO     | backtest.analyze:pipeline:192 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.516820   5.044000  26.301000        21.257000
# SH.513360   2.787000  22.907000        20.120000
# SH.513700   7.889000  27.729000        19.840000
# SZ.159856   4.632000  21.680000        17.048000
# SZ.159828   7.830000  24.393000        16.563000
#              ...        ...              ...
# SH.512600  35.974000  25.105000       -10.869000
# SH.512390  34.168000  22.932000       -11.236000
# SH.510170  35.968000  22.715000       -13.253000
# SH.512690  62.992000  21.464000       -41.528000
# mean       20.402215  23.749702         3.347491
# 
# [663 rows x 3 columns]
# 2024-10-11 21:07:33.601 | INFO     | backtest.analyze:pipeline:196 - rank_personal: 
#        full_code                  comparison_experiment          window  \
# 110    SH.512690                                   base  None-None-None   
# 48310  SH.515700  ablation_macd_diff_dif_dea_0_20241011        13-25-10   
# 34963  SZ.159949  ablation_macd_diff_dif_dea_0_20241011        12-25-10   
# 48743  SZ.159949  ablation_macd_diff_dif_dea_0_20241011        13-25-10   
# 56577  SH.515700  ablation_macd_diff_dif_dea_0_20241011        13-28-10   
#          ...                                    ...             ...   
# 42711  SH.513060  ablation_macd_diff_dif_dea_0_20241011        12-28-10   
# 62003  SH.513060  ablation_macd_diff_dif_dea_0_20241011        14-25-10   
# 156    SH.513330                                   base  None-None-None   
# 56489  SH.513060  ablation_macd_diff_dif_dea_0_20241011        13-28-10   
# 37198  SH.513060  ablation_macd_diff_dif_dea_0_20241011        12-26-10   
# 
#        bar_num  ann_return  max_dd   score  
# 110        892      68.997  49.451  62.992  
# 48310      707      25.281  11.618  43.878  
# 34963      972      29.508  24.797  43.344  
# 48743      972      28.602  24.797  42.760  
# 56577      707      28.733  26.081  42.461  
#        ...         ...     ...     ...  
# 42711      430     -17.443  66.133  -0.030  
# 62003      430     -18.359  64.494  -0.127  
# 156        460     -17.573  67.446  -0.484  
# 56489      430     -19.343  64.746  -0.884  
# 37198      430     -20.148  66.133  -1.798  
# 
# [13420 rows x 7 columns]
# 
# =============================================================================
