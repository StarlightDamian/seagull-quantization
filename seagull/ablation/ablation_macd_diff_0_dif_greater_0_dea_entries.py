# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_0_dif_greater_0_dea_entries
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
            # cache_dict,字典，用于缓存计算结果
        )
        
        # 计算 DIF 和 DEA 的斜率
        dif_slope = macd.macd.diff()  # DIF 线的斜率
        dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 交易信号
        #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
        #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        entries = (dif_slope > 0)  & (dea_slope > 0) & (dea_slope.shift(1) < 0)#& (dif_slope.shift(1) < 0)
        
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
    comparison_experiment = "ablation_macd_diff_0_dif_greater_0_dea_entries_20250205_2"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                               comparison_experiment=comparison_experiment,
                                               )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
# =============================================================================
# 2025-02-05 21:58:13.527 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 5249  2022-08-04 2024-02-02     367      10000.0   8754.949       -12.451   
# 5250  2022-08-04 2024-02-02     367      10000.0  10186.328         1.863   
# 5251  2022-08-04 2024-02-02     367      10000.0   9013.381        -9.866   
# 5252  2022-08-04 2024-02-02     367      10000.0   9011.990        -9.880   
# 5253  2022-08-04 2024-02-02     367      10000.0   8225.480       -17.745   
#          ...        ...     ...          ...        ...           ...   
# 64492 2022-08-04 2024-02-02     367      10000.0  13615.026        36.150   
# 64493 2022-08-04 2024-02-02     367      10000.0  12752.328        27.523   
# 64494 2022-08-04 2024-02-02     367      10000.0   9305.133        -6.949   
# 64495 2022-08-04 2024-02-02     367      10000.0  11800.150        18.002   
# 64496 2022-08-04 2024-02-02     367      10000.0   9155.285        -8.447   
# 
#        benchmark_return  max_dd  bar_num  price_start  price_end  \
# 5249            -13.803  12.451      367        2.927      2.523   
# 5250              6.989   9.426      367        0.744      0.796   
# 5251            -16.031  14.327      367        2.701      2.268   
# 5252              5.951  17.213      367        1.882      1.994   
# 5253            -29.205  20.936      367        0.654      0.463   
#                 ...     ...      ...          ...        ...   
# 64492           -22.427  15.794      367        0.758      0.588   
# 64493           -41.660  18.048      367        1.217      0.710   
# 64494            -7.714  11.517      367        1.076      0.993   
# 64495           -29.471  16.837      367        1.001      0.706   
# 64496           -23.096  34.426      367        0.788      0.606   
# 
#        total_open_trades  total_closed_trades  
# 5249                   1                   11  
# 5250                   1                   10  
# 5251                   1                   11  
# 5252                   1                   11  
# 5253                   0                   14  
#                  ...                  ...  
# 64492                  0                    7  
# 64493                  0                    9  
# 64494                  0                   10  
# 64495                  0                   10  
# 64496                  0                    9  
# 
# [14575 rows x 13 columns]
# 2025-02-05 21:58:13.539 | INFO     | backtest.analyze:pipeline:158 - strategy_total_open_trades_mean: 0.138
# 2025-02-05 21:58:13.544 | INFO     | backtest.analyze:pipeline:159 - strategy_total_closed_trades_mean: 11.234
# 2025-02-05 21:58:13.555 | INFO     | backtest.analyze:pipeline:169 - score_mean_base: 8.958
# 2025-02-05 21:58:13.560 | INFO     | backtest.analyze:pipeline:170 - score_mean_strategy_effective: 20.614
# 2025-02-05 21:58:13.578 | INFO     | backtest.analyze:pipeline:181 - strategy_rank: 
# window
# 12-27-10          21.738
# 12-28-10          21.576
# 10-28-10          21.448
# 12-26-10          21.323
# 13-27-10          21.057
#  
# 12-24-10          19.895
# 14-25-10          19.792
# 14-24-10          19.209
# 13-24-10          19.007
# None-None-None     8.958
# Length: 26, dtype: float64
# 2025-02-05 21:58:13.819 | INFO     | backtest.analyze:pipeline:185 - rank_portfolio: 
# full_code
# SH.513080    36.840
# SH.513030    36.177
# SH.513100    36.002
# SZ.159632    35.706
# SZ.159941    34.768
#  
# SH.561910     5.063
# SZ.159796     5.016
# SZ.159752     4.739
# SH.516270     4.253
# SZ.159967     3.680
# Length: 583, dtype: float64
# 2025-02-05 21:58:13.827 | INFO     | backtest.analyze:pipeline:194 - baseline: 13.439
# 2025-02-05 21:58:13.833 | INFO     | backtest.analyze:pipeline:195 - baseline_strategy: 24.559
# 2025-02-05 21:58:13.967 | INFO     | backtest.analyze:pipeline:201 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.512480  -3.055000  32.708000        35.763000
# SH.516350  -1.638000  29.951000        31.589000
# SH.516920  -2.460000  28.890000        31.350000
# SH.512760  -3.432000  27.642000        31.074000
# SH.516640  -2.348000  27.713000        30.061000
#              ...        ...              ...
# SH.513800  40.352000  28.686000       -11.666000
# SH.513520  40.082000  25.648000       -14.434000
# SZ.159866  41.677000  26.696000       -14.981000
# SH.513360  35.194000  17.059000       -18.135000
# mean        8.957539  20.614036        11.656497
# 
# [584 rows x 3 columns]
# 2025-02-05 21:58:13.988 | INFO     | backtest.analyze:pipeline:205 - rank_personal: 
#        full_code                              comparison_experiment  \
# 120    SH.513300                                               base   
# 110    SH.513100                                               base   
# 554    SZ.159941                                               base   
# 52026  SZ.159997  ablation_macd_diff_0_dif_greater_0_dea_entries...   
# 51474  SH.512480  ablation_macd_diff_0_dif_greater_0_dea_entries...   
#          ...                                                ...   
# 445    SZ.159775                                               base   
# 434    SZ.159755                                               base   
# 347    SH.562880                                               base   
# 323    SH.561160                                               base   
# 461    SZ.159796                                               base   
# 
#                window  bar_num  ann_return  max_dd   score  
# 120    None-None-None      367      26.203  14.739  43.638  
# 110    None-None-None      367      25.282  16.136  42.645  
# 554    None-None-None      367      24.773  16.667  42.112  
# 52026        13-28-10      367      23.101  14.260  41.907  
# 51474        13-28-10      367      23.838  16.665  41.745  
#               ...      ...         ...     ...     ...  
# 445    None-None-None      367     -44.745  60.600 -17.761  
# 434    None-None-None      367     -44.734  60.621 -17.835  
# 347    None-None-None      367     -44.404  61.804 -17.890  
# 323    None-None-None      367     -44.924  62.090 -18.336  
# 461    None-None-None      367     -45.011  62.324 -18.457  
# 
# [15158 rows x 7 columns]
# 
# =============================================================================
