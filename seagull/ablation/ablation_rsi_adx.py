# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:12:35 2025

@author: awei
(ablation_rsi_adx)
"""

import os
import itertools

import vectorbt as vbt
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_log, utils_character
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class ablationMacd(vectorbt_macd.backtestVectorbtMacd, analyze.backtestAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macd_hist = None
    
    def strategy(self, subtable_df, data):
        window_fast, window_slow, window_signal = subtable_df[['window_fast', 'window_slow', 'window_signal']].values[0]
        
        global data1
        data1 = data
        # https://github.com/polakowo/vectorbt/blob/54cbe7c5bff332b510d1075c5cf11d006c1b1846/vectorbt/indicators/nb.py#L171
        macd = vbt.MACD.run(
            close=data,  # close: 2D数组，表示收盘价
            fast_window=window_fast,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
            slow_window=window_slow,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
            signal_window=window_signal,  # 信号线的窗口大小,Signal line period, default value 9,这个参数好像没什么用
            macd_ewm=False,  # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
            signal_ewm=True,  # 布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
            adjust=False #布尔值，是否在计算EMA时进行调整
            #cache_dict,字典，用于缓存计算结果
        )
        #df['adx'] = talib.ADX(high, low, close, timeperiod=14)

        rsi = vbt.RSI.run(data, window=14).rsi
        
        #dif_slope = macd.macd.diff()  # DIF 线的斜率
        #dea_slope = macd.signal.diff()  # DEA 线的斜率
        #entries1 = (dif_slope > 0) & (dif_slope.shift(1) < 0) & (dea_slope > 0) & (dea_slope.shift(1) < 0)
        #entries1.columns = entries1.columns.droplevel([0, 1, 2, 3, 4])  # 删除前5个层级

        #entries2 = rsi < 15  # 买入信号：RSI < 30
        #entries2.columns = entries2.columns.droplevel([0])  # 删除前5个层级

        #entries = entries1 | entries2
        #exits = rsi > 40  # 卖出信号：RSI > 70
        #exits.columns = exits.columns.droplevel([0])  # 删除前5个层级
        
        #entries = rsi < 15  # 买入信号：RSI < 30
        #exits = rsi > 40  # 卖出信号：RSI > 70
        entries = rsi < 15  # 买入信号：RSI < 30
        exits = rsi > 40  # 卖出信号：RSI > 70
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
    comparison_experiment = utils_character.generate_random_string(6)  #"ablation_macd_diff_dif_dea_entries_1_20250205_27"
    ablation_macd.ablation_experiment(date_start='2022-08-01',  # '2019-01-01', 
                                      date_end='2024-02-02',  # '2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)
    
    # output
    eval_columns = ['full_code','ann_return','max_dd','calmar_ratio',
                    'total_closed_trades','win_rate','profit_loss_ratio']
    base_df[eval_columns].to_csv(f'{PATH}/data/eval_base.csv', index=False)
    strategy_df[eval_columns].to_csv(f'{PATH}/data/eval_strategy.csv', index=False)
    
# =============================================================================
# import vectorbt as vbt
# import pandas as pd
# 
# # 1. 下载数据（假设是从 Yahoo Finance 下载）
# symbols = ["ADA-USD", "ETH-USD"]
# price_data = vbt.YFData.download(symbols, start='2018-01-01', end='2022-12-31').get('Close')
# 
# # 2. 计算 RSI 指标
# #rsi = price_data.vbt.RSI(window=14)  # 14 日 RSI
# rsi = vbt.RSI.run(price_data, window=14).rsi
# 
# # 3. 定义买卖信号：假设 RSI < 30 时买入，RSI > 70 时卖出
# 
# entries = rsi < 30  # 买入信号：RSI < 30
# exits = rsi > 70  # 卖出信号：RSI > 70
# 
# # 4. 运行回测
# portfolio = vbt.Portfolio.from_signals(price_data, entries, exits, freq='1D')
# 
# # 5. 显示结果
# portfolio.total_return()
# portfolio.plot()
# 
# =============================================================================

# =============================================================================
# 2025-02-07 15:28:56.420 | INFO     | backtest.analyze:pipeline:156 -       date_start   date_end  period  start_value  end_value  total_return  \
# 38899 2022-08-04 2024-02-02     367      10000.0  10126.693         1.267   
# 38901 2022-08-04 2024-02-02     367      10000.0   9881.850        -1.181   
# 38903 2022-08-04 2024-02-02     367      10000.0  10151.922         1.519   
# 38904 2022-08-04 2024-02-02     367      10000.0  10421.637         4.216   
# 38905 2022-08-04 2024-02-02     367      10000.0  10969.708         9.697   
#          ...        ...     ...          ...        ...           ...   
# 39476 2022-08-04 2024-02-02     367      10000.0  10154.547         1.545   
# 39477 2022-08-04 2024-02-02     367      10000.0   9762.097        -2.379   
# 39478 2022-08-04 2024-02-02     367      10000.0  10604.451         6.045   
# 39480 2022-08-04 2024-02-02     367      10000.0  10389.797         3.898   
# 39481 2022-08-04 2024-02-02     367      10000.0  10552.506         5.525   
# 
#        benchmark_return  ann_return  max_dd  calmar_ratio  bar_num  \
# 38899           -13.803       0.837   3.428         0.244      367   
# 38901           -16.031      -0.784  11.455        -0.068      367   
# 38903           -29.205       1.003   3.834         0.262      367   
# 38904             0.881       2.772   1.156         2.398      367   
# 38905           -15.034       6.320   2.171         2.911      367   
#                 ...         ...     ...           ...      ...   
# 39476            -4.941       1.021   1.963         0.520      367   
# 39477           -22.427      -1.582   8.967        -0.176      367   
# 39478           -41.660       3.962   9.454         0.419      367   
# 39480           -29.471       2.564   7.328         0.350      367   
# 39481           -23.096       3.625   1.793         2.021      367   
# 
#        price_start  price_end  total_open_trades  total_closed_trades  
# 38899        2.927      2.523                  0                    2  
# 38901        2.701      2.268                  0                    2  
# 38903        0.654      0.463                  0                    2  
# 38904        0.681      0.687                  0                    2  
# 38905        0.878      0.746                  0                    3  
#            ...        ...                ...                  ...  
# 39476        0.931      0.885                  0                    2  
# 39477        0.758      0.588                  0                    1  
# 39478        1.217      0.710                  0                    3  
# 39480        1.001      0.706                  0                    2  
# 39481        0.788      0.606                  0                    2  
# 
# [509 rows x 15 columns]
# 2025-02-07 15:28:56.434 | INFO     | backtest.analyze:pipeline:165 - base_ann_return_mean: -16.608
# 2025-02-07 15:28:56.436 | INFO     | backtest.analyze:pipeline:166 - strategy_ann_return_mean: 0.434
# 2025-02-07 15:28:56.437 | INFO     | backtest.analyze:pipeline:167 - base_max_dd_mean: 33.459
# 2025-02-07 15:28:56.438 | INFO     | backtest.analyze:pipeline:168 - strategy_max_dd_mean: 7.161
# 2025-02-07 15:28:56.439 | INFO     | backtest.analyze:pipeline:169 - strategy_total_open_trades_mean: 0.161
# 2025-02-07 15:28:56.441 | INFO     | backtest.analyze:pipeline:170 - strategy_total_closed_trades_mean: 2.275
# 2025-02-07 15:28:56.442 | INFO     | backtest.analyze:pipeline:175 - score_mean_base: 8.958
# 2025-02-07 15:28:56.443 | INFO     | backtest.analyze:pipeline:176 - score_mean_strategy_effective: 28.222
# 2025-02-07 15:28:56.446 | INFO     | backtest.analyze:pipeline:187 - strategy_rank: 
# window
# None-None-None    14.812
# dtype: float64
# 2025-02-07 15:28:56.515 | INFO     | backtest.analyze:pipeline:191 - rank_portfolio: 
# full_code
# SH.513300    43.638
# SH.513100    42.645
# SZ.159941    42.112
# SZ.159632    41.328
# SH.513080    40.620
#  
# SZ.159767    -5.091
# SZ.159757    -5.335
# SZ.159755    -5.536
# SZ.159796    -5.656
# SZ.159775    -5.816
# Length: 583, dtype: float64
# 2025-02-07 15:28:56.517 | INFO     | backtest.analyze:pipeline:200 - baseline: 13.439
# 2025-02-07 15:28:56.519 | INFO     | backtest.analyze:pipeline:201 - baseline_strategy: 30.377
# 2025-02-07 15:28:56.646 | INFO     | backtest.analyze:pipeline:207 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# SH.516880 -15.199000  34.381000        49.580000
# SH.515790 -15.958000  33.387000        49.345000
# SH.516180 -15.294000  33.851000        49.145000
# SZ.159857 -15.675000  33.346000        49.021000
# SZ.159618 -15.769000  33.178000        48.947000
#              ...        ...              ...
# SH.517090  37.563000  31.217000        -6.346000
# SH.517180  40.480000  31.783000        -8.697000
# SH.513030  41.691000  30.852000       -10.839000
# SZ.159866  41.677000  29.811000       -11.866000
# mean        6.960972  28.221668        21.260695
# 
# [510 rows x 3 columns]
# 2025-02-07 15:28:56.653 | INFO     | backtest.analyze:pipeline:211 - rank_personal: 
#       full_code comparison_experiment          window  bar_num  ann_return  \
# 541   SH.513300                  base  None-None-None      367      26.203   
# 4039  SH.513300                  base  None-None-None      367      26.203   
# 4029  SH.513100                  base  None-None-None      367      25.282   
# 531   SH.513100                  base  None-None-None      367      25.282   
# 4473  SZ.159941                  base  None-None-None      367      24.773   
#         ...                   ...             ...      ...         ...   
# 185   SH.562880                  base  None-None-None      367     -44.404   
# 161   SH.561160                  base  None-None-None      367     -44.924   
# 4242  SH.561160                  base  None-None-None      367     -44.924   
# 4380  SZ.159796                  base  None-None-None      367     -45.011   
# 299   SZ.159796                  base  None-None-None      367     -45.011   
# 
#       max_dd   score  
# 541   14.739  43.638  
# 4039  14.739  43.638  
# 4029  16.136  42.645  
# 531   16.136  42.645  
# 4473  16.667  42.112  
#      ...     ...  
# 185   61.804 -17.890  
# 161   62.090 -18.336  
# 4242  62.090 -18.336  
# 4380  62.324 -18.457  
# 299   62.324 -18.457  
# 
# [1675 rows x 7 columns]
# =============================================================================

df['adx'] = talib.ADX(high, low, close, timeperiod=14)


