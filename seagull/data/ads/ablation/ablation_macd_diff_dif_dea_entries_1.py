# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_diff_dif_dea_entries_1
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
        #dif_slope = macd.macd.diff()  # DIF 线的斜率
        #dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 交易信号
        #entries = (dif_slope > dea_slope) # 买入条件：DIF的斜率开始大于DEA的斜率
        #exits = (dif_slope < 0) # 卖出条件：DIF斜率=0
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        #entries = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0)# & (dea_slope.shift(1) < 0)
        
        # 卖出信号：DIF 的斜率从正变负
        #exits = (dif_slope < 0) & (dif_slope.shift(1) > 0)
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
    comparison_experiment = "ablation_macd_diff_dif_dea_entries_1_20250205_6"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    
# =============================================================================
# 2025-02-05 22:00:07.994 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 13230 2022-08-04 2024-02-02     367      10000.0   9952.693        -0.473   
# 13231 2022-08-04 2024-02-02     367      10000.0   8886.881       -11.131   
# 13232 2022-08-04 2024-02-02     367      10000.0  10429.342         4.293   
# 13233 2022-08-04 2024-02-02     367      10000.0  11218.911        12.189   
# 13234 2022-08-04 2024-02-02     367      10000.0   9147.935        -8.521   
#          ...        ...     ...          ...        ...           ...   
# 13808 2022-08-04 2024-02-02     367      10000.0   9349.501        -6.505   
# 13809 2022-08-04 2024-02-02     367      10000.0   9394.356        -6.056   
# 13810 2022-08-04 2024-02-02     367      10000.0  10124.137         1.241   
# 13811 2022-08-04 2024-02-02     367      10000.0   8855.965       -11.440   
# 13812 2022-08-04 2024-02-02     367      10000.0   9850.376        -1.496   
# 
#        benchmark_return  max_dd  bar_num  price_start  price_end  \
# 13230           -13.803   5.243      367        2.927      2.523   
# 13231             6.989  11.131      367        0.744      0.796   
# 13232           -16.031   1.226      367        2.701      2.268   
# 13233             5.951   5.275      367        1.882      1.994   
# 13234           -29.205   9.690      367        0.654      0.463   
#                 ...     ...      ...          ...        ...   
# 13808           -22.427   6.505      367        0.758      0.588   
# 13809           -41.660   7.048      367        1.217      0.710   
# 13810            -7.714   6.636      367        1.076      0.993   
# 13811           -29.471  12.234      367        1.001      0.706   
# 13812           -23.096  20.612      367        0.788      0.606   
# 
#        total_open_trades  total_closed_trades  
# 13230                  1                    4  
# 13231                  0                    9  
# 13232                  0                    1  
# 13233                  0                    8  
# 13234                  0                    5  
#                  ...                  ...  
# 13808                  0                    2  
# 13809                  0                    8  
# 13810                  0                    7  
# 13811                  0                    9  
# 13812                  1                    7  
# 
# [583 rows x 13 columns]
# 2025-02-05 22:00:08.006 | INFO     | backtest.analyze:pipeline:158 - strategy_total_open_trades_mean: 0.123
# 2025-02-05 22:00:08.012 | INFO     | backtest.analyze:pipeline:159 - strategy_total_closed_trades_mean: 6.779
# 2025-02-05 22:00:08.020 | INFO     | backtest.analyze:pipeline:169 - score_mean_base: 8.958
# 2025-02-05 22:00:08.026 | INFO     | backtest.analyze:pipeline:170 - score_mean_strategy_effective: 25.425
# 2025-02-05 22:00:08.034 | INFO     | backtest.analyze:pipeline:181 - strategy_rank: 
# window
# 12-26-9           25.425
# None-None-None     8.958
# dtype: float64
# 2025-02-05 22:00:08.104 | INFO     | backtest.analyze:pipeline:185 - rank_portfolio: 
# full_code
# SH.513520    36.537
# SZ.159866    35.679
# SZ.159612    34.712
# SH.513800    34.441
# SZ.159941    34.414
#  
# SZ.159752    -2.873
# SH.516090    -3.026
# SH.561160    -3.475
# SZ.159796    -4.029
# SZ.159840    -5.007
# Length: 583, dtype: float64
# 2025-02-05 22:00:08.111 | INFO     | backtest.analyze:pipeline:194 - baseline: 13.439
# 2025-02-05 22:00:08.117 | INFO     | backtest.analyze:pipeline:195 - baseline_strategy: 27.773
# 2025-02-05 22:00:08.241 | INFO     | backtest.analyze:pipeline:201 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.561800 -12.867000  30.681000        43.548000
# SZ.159608 -12.056000  28.387000        40.443000
# SZ.159636  -0.760000  39.649000        40.409000
# SH.562800 -12.332000  25.808000        38.140000
# SZ.159703 -13.469000  24.261000        37.730000
#              ...        ...              ...
# SH.513030  41.691000  25.974000       -15.717000
# SH.513080  40.620000  24.749000       -15.871000
# SH.513100  42.645000  24.857000       -17.788000
# SH.513300  43.638000  24.092000       -19.546000
# mean        8.957539  25.424557        16.467019
# 
# [584 rows x 3 columns]
# 2025-02-05 22:00:08.251 | INFO     | backtest.analyze:pipeline:205 - rank_personal: 
#        full_code                            comparison_experiment  \
# 120    SH.513300                                             base   
# 110    SH.513100                                             base   
# 13332  SH.513010  ablation_macd_diff_dif_dea_entries_1_20250205_6   
# 554    SZ.159941                                             base   
# 104    SH.513030                                             base   
#          ...                                              ...   
# 445    SZ.159775                                             base   
# 434    SZ.159755                                             base   
# 347    SH.562880                                             base   
# 323    SH.561160                                             base   
# 461    SZ.159796                                             base   
# 
#                window  bar_num  ann_return  max_dd   score  
# 120    None-None-None      367      26.203  14.739  43.638  
# 110    None-None-None      367      25.282  16.136  42.645  
# 13332         12-26-9      367      21.555   7.393  42.454  
# 554    None-None-None      367      24.773  16.667  42.112  
# 104    None-None-None      367      22.893  13.654  41.691  
#               ...      ...         ...     ...     ...  
# 445    None-None-None      367     -44.745  60.600 -17.761  
# 434    None-None-None      367     -44.734  60.621 -17.835  
# 347    None-None-None      367     -44.404  61.804 -17.890  
# 323    None-None-None      367     -44.924  62.090 -18.336  
# 461    None-None-None      367     -45.011  62.324 -18.457  
# 
# [1166 rows x 7 columns]
# =============================================================================
