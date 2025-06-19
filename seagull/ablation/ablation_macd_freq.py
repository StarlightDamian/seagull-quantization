# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:15:40 2024

@author: awei
ablation_macd_freq
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
import efinance as ef
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from __init__ import path
from utils import utils_log, utils_database  # , utils_data, utils_character
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


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
#             macd_ewm=False,  # #布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
#             signal_ewm=True, #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
#             adjust=False #布尔值，是否在计算EMA时进行调整
#             #cache_dict,字典，用于缓存计算结果
#         )
#         
#         self.macd_hist = macd.hist # macd能量柱
#         
# 
#         # cache_dict 示例数据
#         # 用于存储预先计算的移动平均线，以提高性能
#         # 创建缓存字典（这里只是示例，实际使用时需要预先计算）
#         # cache_dict = {
#         #     hash((fast_window, macd_ewm)): np.random.randn(100, 2),  # 模拟快速MA
#         #     hash((slow_window, macd_ewm)): np.random.randn(100, 2)   # 模拟慢速MA
#         # }
#         
#         # https://github.com/polakowo/vectorbt/issues/136
#         entries = macd.macd_above(0) & macd.macd_below(0).vbt.signals.fshift(1)
#         exits = macd.macd_below(0) & macd.macd_above(0).vbt.signals.fshift(1)
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
    
    def dataset(self, symbols,
                date_start='20190101',
                date_end='20230101',):
        #date_start='20240821'#'20190101'
        date_start='20240821'
        date_end='20240920'
        #date_end='20241011'#'20230101'
        #date_start='20190101'
        #date_end='20230101'
        klt=5
        with utils_database.engine_conn('postgre') as conn:
            dwd_portfolio_base = pd.read_sql("dwd_info_nrtd_portfolio_base", con=conn.engine)
        dwd_portfolio_base = dwd_portfolio_base[~(dwd_portfolio_base.prev_close=='-')]
        etf_dict = dict(zip(dwd_portfolio_base['asset_code'], dwd_portfolio_base['code_name']))

        etf_freq_dict = ef.stock.get_quote_history(list(etf_dict.keys()),
                                                   klt=str(klt),
                                                   beg=date_start,
                                                   end=date_end
                                                   )
        # etf_freq_df = pd.concat(etf_freq_dict, names=['code', 'row']).reset_index(level='row', drop=True)
        etf_freq_df = pd.concat({k: v for k, v in etf_freq_dict.items()}).reset_index(drop=True)
        etf_freq_df = etf_freq_df.rename(columns={'日期': 'date',
                                                  '股票代码': 'full_code',
                                                  '收盘': 'close'})
        etf_freq_df['date'] = pd.to_datetime(etf_freq_df['date'])
        if klt==103:
            etf_freq_df['date'] = pd.to_datetime(etf_freq_df['date'].dt.strftime('%Y-%m-01'))
            #处理月线数据时，问题可能出现在数据的时间频率不匹配或数据不完整上。如果你在日线数据上做透视（pivot）操作是没有问题的，但处理月线数据时感觉异常
        #    etf_freq_df = etf_freq_df.set_index('date').resample('M').last().reset_index()
        #elif klt==102:
        #    etf_freq_df = etf_freq_df.set_index('date').resample('W').last().reset_index()
        etf_freq_df = etf_freq_df.pivot(index='date', columns='full_code', values='close')
        
        etf_freq_df.columns.name = 'symbol'
        #vbt.Portfolio.from_signals()会对数据自动补全，因此全空数据error:TypingError
        etf_freq_df = etf_freq_df.loc[:,~(etf_freq_df.count()==0)]
        global etf_freq_dict1
        etf_freq_dict1=etf_freq_dict
        return etf_freq_df
    
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
                                 strategy_params_batch_size=256,  # MemoryError
                                 portfolio_params={'freq': '5min',
                                                   'fees': 0.001,  # 0.1% per trade
                                                   'slippage': 0.001,  # 0.1% slippage
                                                   'init_cash': 10000},
                                 strategy_params_list=strategy_params_list,
                                 )
    
    dataset = ablation_macd.dataset(symbols=5,
                                    date_start='2019-01-01',
                                    date_end='2023-01-01',)
    comparison_experiment = "ablation_macd_freq_5_2"
    ablation_macd.ablation_experiment(date_start='2019-01-01',
                                      date_end='2023-01-01',
                                      comparison_experiment=comparison_experiment,
                                      )
    bacetest_df, base_df, strategy_df = ablation_macd.pipeline(comparison_experiment=comparison_experiment)

# =============================================================================
# 101
# 2024-10-12 20:49:30.598 | INFO     | backtest.analyze:pipeline:163 - score_mean_base: 20.576
# 2024-10-12 20:49:30.601 | INFO     | backtest.analyze:pipeline:164 - score_mean_strategy_effective: 27.707
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:174: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-12 20:49:30.612 | INFO     | backtest.analyze:pipeline:175 - strategy_rank: 
# window
# 10-24-10          28.390
# 10-28-10          28.319
# 10-27-10          28.228
# 10-25-10          28.201
# 12-26-10          28.188
#  
# 13-27-10          27.577
# 11-25-10          27.555
# 12-25-7           27.252
# 14-27-10          27.241
# None-None-None    20.576
# Length: 28, dtype: float64
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:178: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-12 20:49:30.738 | INFO     | backtest.analyze:pipeline:179 - rank_portfolio: 
# full_code
# SH.512690    63.031
# SH.512520    37.820
# SH.515220    37.273
# SH.512040    37.158
# SH.512600    35.986
#  
# SZ.159740     3.364
# SH.513130     3.019
# SZ.159747     2.934
# SH.513360     2.781
# SH.513330    -0.492
# Length: 1347, dtype: float64
# 2024-10-12 20:49:30.746 | INFO     | backtest.analyze:pipeline:187 - baseline: 26.089
# 2024-10-12 20:49:30.746 | INFO     | backtest.analyze:pipeline:188 - baseline_strategy: nan
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:33: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
# 2024-10-12 20:49:30.914 | INFO     | backtest.analyze:pipeline:192 - comparison_portfolio: 
#                 base   strategy  strategy_better
# full_code                                       
# 513330     -0.484000  26.666000        27.150000
# 513360      2.787000  27.952000        25.165000
# 513130      3.025000  27.770000        24.745000
# 513980      3.602000  27.921000        24.319000
# 513580      3.625000  27.910000        24.285000
#              ...        ...              ...
# 512600     35.974000  27.765000        -8.209000
# 512520     37.808000  28.143000        -9.665000
# 512390     36.334000  26.087000       -10.247000
# 512690     62.992000  25.686000       -37.306000
# mean       20.255807  28.026513         7.770702
# 
# [638 rows x 3 columns]
# 2024-10-12 20:49:30.925 | INFO     | backtest.analyze:pipeline:196 - rank_personal: 
#        full_code comparison_experiment          window  bar_num  ann_return  \
# 64433  SH.512690                  base  None-None-None     1335      69.058   
# 92137     512690                  base  None-None-None      892      68.997   
# 64421  SH.512520                  base  None-None-None     1459      24.641   
# 92107     512520                  base  None-None-None      972      24.622   
# 64521  SH.515220                  base  None-None-None     1034      25.549   
#          ...                   ...             ...      ...         ...   
# 64840  SZ.159747                  base  None-None-None      535     -14.873   
# 93559     513360                  base  None-None-None      377     -12.543   
# 64480  SH.513360                  base  None-None-None      562     -12.551   
# 93557     513330                  base  None-None-None      460     -17.573   
# 64479  SH.513330                  base  None-None-None      691     -17.584   
# 
#        max_dd   score  
# 64433  49.451  63.031  
# 92137  49.451  62.992  
# 64421  32.843  37.820  
# 92107  32.843  37.808  
# 64521  35.489  37.273  
#       ...     ...  
# 64840  61.561   2.934  
# 93559  68.663   2.787  
# 64480  68.663   2.781  
# 93557  67.446  -0.484  
# 64479  67.446  -0.492  
# 
# [13688 rows x 7 columns]
# =============================================================================

# =============================================================================
# 103
# comparison_experiment = "ablation_macd_freq_103_2"
# 2024-10-13 14:38:58.786 | INFO     | backtest.analyze:pipeline:154 -                      start                  end  period  start_value  \
# 4388   2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 4602   2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 4686   2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 4688   2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 4689   2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
#                    ...                  ...     ...          ...   
# 66134  2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 66146  2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 66159  2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 66177  2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 66180  2019-01-01 00:00:00  2022-12-01 00:00:00      48      10000.0   
# 
#        end_value  total_return  benchmark_return  max_gross_exposure  \
# 4388    9959.283        -0.407             0.003               100.0   
# 4602    8570.587       -14.294            60.428               100.0   
# 4686    9054.075        -9.459             0.185               100.0   
# 4688    9866.503        -1.335             0.018               100.0   
# 4689    9865.322        -1.347             0.020               100.0   
#          ...           ...               ...                 ...   
# 66134   9910.203        -0.898             0.011               100.0   
# 66146   8394.303       -16.057            -2.803               100.0   
# 66159   8941.283       -10.587            84.726               100.0   
# 66177   7702.543       -22.975           101.165               100.0   
# 66180  10007.284         0.073            60.707               100.0   
# 
#        total_fees_paid  max_dd  ... slippage  freq_portfolio  bar_num  \
# 4388            19.959   0.407  ...    0.001              MS       48   
# 4602            18.569  14.545  ...    0.001              MS       48   
# 4686            19.053   9.459  ...    0.001              MS       48   
# 4688            59.592   1.335  ...    0.001              MS       48   
# 4689            59.576   1.347  ...    0.001              MS       48   
#                ...     ...  ...      ...             ...      ...   
# 66134           39.809   0.898  ...    0.001              MS       48   
# 66146           18.393  16.057  ...    0.001              MS       48   
# 66159           18.940  10.587  ...    0.001              MS       48   
# 66177           17.700  22.975  ...    0.001              MS       48   
# 66180           20.007   0.200  ...    0.001              MS       48   
# 
#        date_start   date_end  price_start  price_end  \
# 4388   2019-01-01 2022-12-01      100.000    100.003   
# 4602   2019-01-01 2022-12-01        0.374      0.600   
# 4686   2019-01-01 2022-12-01       99.933    100.118   
# 4688   2019-01-01 2022-12-01       99.997    100.015   
# 4689   2019-01-01 2022-12-01       99.991    100.011   
#           ...        ...          ...        ...   
# 66134  2019-01-01 2022-12-01      100.003    100.014   
# 66146  2019-01-01 2022-12-01        2.069      2.011   
# 66159  2019-01-01 2022-12-01        0.347      0.641   
# 66177  2019-01-01 2022-12-01        0.601      1.209   
# 66180  2019-01-01 2022-12-01        0.509      0.818   
# 
#                             primary_key     insert_timestamp    window  
# 4388   2ebdd41e150783a6b2a69869cbd0581e  2024-10-13 14:38:40  10-24-10  
# 4602   0acf0573bec39b48e14bb8cb6c41e251  2024-10-13 14:38:40  10-24-10  
# 4686   1f4909faf65159ede9273f0292da9e29  2024-10-13 14:38:40  10-24-10  
# 4688   95a929e9c97ed8881d163406e4da0761  2024-10-13 14:38:40  10-24-10  
# 4689   fba7274635fbe05259079fdfe91f37fb  2024-10-13 14:38:40  10-24-10  
#                                 ...                  ...       ...  
# 66134  7df8be68105c25525ed4ef5e2ee5c6fb  2024-10-13 14:38:40   10-25-8  
# 66146  6cb5d21c6bd812302705294155e4e5b7  2024-10-13 14:38:40   10-25-8  
# 66159  f25706e0fcd5989b2c34cfb37c80e434  2024-10-13 14:38:40   10-25-8  
# 66177  d3209714e324a40871445fd72c3080bb  2024-10-13 14:38:40   10-25-8  
# 66180  c58c23523e07a256874be37940f616e6  2024-10-13 14:38:40   10-25-8  
# 
# [282 rows x 56 columns]
# 2024-10-13 14:38:58.800 | INFO     | backtest.analyze:pipeline:164 - score_mean_base: 1639.760
# 2024-10-13 14:38:58.802 | INFO     | backtest.analyze:pipeline:165 - score_mean_strategy_effective: 22.996
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:175: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 14:38:58.812 | INFO     | backtest.analyze:pipeline:176 - strategy_rank: 
# window
# None-None-None    269.616
# 14-25-10           28.475
# 12-27-10           28.243
# 12-26-10           28.159
# 12-25-10           27.730
#   
# 10-24-10           19.008
# 10-25-8            15.782
# 11-25-10           15.535
# 10-26-7            14.949
# 10-26-8           -13.214
# Length: 27, dtype: float64
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:179: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 14:38:58.924 | INFO     | backtest.analyze:pipeline:180 - rank_portfolio: 
# full_code
# 512690    164821.452
# 515220      2405.515
# 159949      1249.589
# 159806      1128.659
# 510630      1027.320
#    
# 159740       -24.440
# 159741       -24.791
# 513360       -24.844
# 513130       -25.165
# 513330       -29.872
# Length: 1347, dtype: float64
# 2024-10-13 14:38:58.926 | INFO     | backtest.analyze:pipeline:192 - IndexError: index 0 is out of bounds for axis 0 with size 0
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:33: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
# 2024-10-13 14:38:58.983 | INFO     | backtest.analyze:pipeline:196 - comparison_portfolio: 
#                   base   strategy  strategy_better
# full_code                                         
# 515880      -30.606000  14.402000        45.008000
# 159005       29.276000  29.226000        -0.050000
# 159003       29.263000  28.615000        -0.648000
# 511600       28.933000  28.142000        -0.791000
# 511690       29.264000  28.206000        -1.058000
#                ...        ...              ...
# 512100     1144.388000  -2.676000     -1147.064000
# 512560     1493.554000  -4.093000     -1497.647000
# 512810     2329.573000 -22.696000     -2352.269000
# 511880             NaN  29.434000              NaN
# mean        348.799885  16.539111      -332.756731
# 
# [28 rows x 3 columns]
# 2024-10-13 14:38:58.992 | INFO     | backtest.analyze:pipeline:200 - rank_personal: 
#      full_code comparison_experiment          window  bar_num   ann_return  \
# 3414    512690                  base  None-None-None       44  1330201.424   
# 110     512690                  base  None-None-None       44    81374.850   
# 3499    515220                  base  None-None-None       34    16525.856   
# 3287    159949                  base  None-None-None       48     8103.964   
# 3187    159806                  base  None-None-None       34     7256.258   
#        ...                   ...             ...      ...          ...   
# 3144    159740                  base  None-None-None       20      -95.203   
# 3145    159741                  base  None-None-None       19      -95.639   
# 3446    513130                  base  None-None-None       19      -95.798   
# 3459    513330                  base  None-None-None       23      -96.922   
# 3460    513360                  base  None-None-None       19      -94.567   
# 
#       max_dd       score  
# 3414  42.630  931148.225  
# 110   42.630   56971.429  
# 3499  22.222   11585.896  
# 3287  36.906    5686.483  
# 3187  37.260    5093.172  
#      ...         ...  
# 3144  61.084     -57.167  
# 3145  60.793     -57.301  
# 3446  61.390     -57.697  
# 3459  61.774     -57.949  
# 3460  65.844     -59.407  
# 
# [4915 rows x 7 columns]
# =============================================================================

# =============================================================================
# date_start='20240821'#'20190101'
# date_end='20241011'#'20230101'
# klt=60
# 
# 2024-10-13 18:11:30.861 | INFO     | backtest.analyze:pipeline:164 - score_mean_base: 54.110
# 2024-10-13 18:11:30.863 | INFO     | backtest.analyze:pipeline:165 - score_mean_strategy_effective: 45.451
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:175: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:11:30.876 | INFO     | backtest.analyze:pipeline:176 - strategy_rank: 
# window
# None-None-None    234.204
# 11-28-10           53.132
# 10-27-10           50.828
# 10-26-10           50.466
# 12-27-10           50.354
#   
# 12-25-8            40.922
# 14-27-10           40.655
# 14-25-10           40.346
# 11-24-10           40.072
# 14-24-10           37.461
# Length: 28, dtype: float64
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:179: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:11:31.007 | INFO     | backtest.analyze:pipeline:180 - rank_portfolio: 
# full_code
# 512690    141282.217
# 515220      1460.171
# 512600       722.744
# 159949       672.754
# 512520       648.601
#    
# 515920        -1.132
# 159883        -1.621
# 159898        -4.126
# 159856        -5.856
# 517200        -6.040
# Length: 1599, dtype: float64
# 2024-10-13 18:11:31.010 | INFO     | backtest.analyze:pipeline:192 - IndexError: index 0 is out of bounds for axis 0 with size 0
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:33: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
# 2024-10-13 18:11:31.150 | INFO     | backtest.analyze:pipeline:196 - comparison_portfolio: 
#                  base    strategy  strategy_better
# full_code                                         
# 516920      69.590000  112.824000        43.234000
# 159599      67.943000  109.355000        41.412000
# 159310      69.891000  110.965000        41.074000
# 516350      69.753000  110.349000        40.596000
# 159665      70.137000  110.453000        40.316000
#               ...         ...              ...
# 159923      99.765000   27.665000       -72.100000
# 516860     149.347000   67.824000       -81.523000
# 513090     126.331000   44.544000       -81.788000
# 516100     139.131000   55.397000       -83.734000
# mean        54.470758   47.254367        -7.216385
# 
# [770 rows x 3 columns]
# 2024-10-13 18:11:31.170 | INFO     | backtest.analyze:pipeline:200 - rank_personal: 
#        full_code comparison_experiment          window  bar_num   ann_return  \
# 192950    512690                  base  None-None-None       44  1330201.424   
# 185147    512690                  base  None-None-None       44    81374.850   
# 193035    515220                  base  None-None-None       34    16525.856   
# 193482    159949                  base  None-None-None       48     8103.964   
# 193382    159806                  base  None-None-None       34     7256.258   
#          ...                   ...             ...      ...          ...   
# 193339    159740                  base  None-None-None       20      -95.203   
# 193340    159741                  base  None-None-None       19      -95.639   
# 192982    513130                  base  None-None-None       19      -95.798   
# 192995    513330                  base  None-None-None       23      -96.922   
# 192996    513360                  base  None-None-None       19      -94.567   
# 
#         max_dd       score  
# 192950  42.630  931148.225  
# 185147  42.630   56971.429  
# 193035  22.222   11585.896  
# 193482  36.906    5686.483  
# 193382  37.260    5093.172  
#        ...         ...  
# 193339  61.084     -57.167  
# 193340  60.793     -57.301  
# 192982  61.390     -57.697  
# 192995  61.774     -57.949  
# 192996  65.844     -59.407  
# 
# [11192 rows x 7 columns]
# =============================================================================

# =============================================================================
# 30min
# 2024-10-13 18:19:44.624 | INFO     | backtest.analyze:pipeline:164 - score_mean_base: 38.780
# 2024-10-13 18:19:44.626 | INFO     | backtest.analyze:pipeline:165 - score_mean_strategy_effective: 31.312
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:175: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:19:44.642 | INFO     | backtest.analyze:pipeline:176 - strategy_rank: 
# window
# None-None-None    216.073
# 12-28-10           33.412
# 13-28-10           32.723
# 12-27-10           32.699
# 14-28-10           32.676
#   
# 10-26-10           30.529
# 10-27-10           30.475
# 10-24-10           30.358
# 13-24-10           30.257
# 14-24-10           30.039
# Length: 26, dtype: float64
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:179: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:19:44.798 | INFO     | backtest.analyze:pipeline:180 - rank_portfolio: 
# full_code
# 512690       47121.459
# 159806         667.982
# 515030         566.624
# 159949         542.842
# 515220         478.996
#    
# SZ.159740        3.364
# SH.513130        3.019
# SZ.159747        2.934
# SH.513360        2.781
# SH.513330       -0.492
# Length: 1599, dtype: float64
# 2024-10-13 18:19:44.803 | INFO     | backtest.analyze:pipeline:192 - IndexError: index 0 is out of bounds for axis 0 with size 0
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:33: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
# 2024-10-13 18:19:44.956 | INFO     | backtest.analyze:pipeline:196 - comparison_portfolio: 
#                base   strategy  strategy_better
# full_code                                      
# 588860     33.73000  42.954000         9.224000
# 159873     37.48600  45.631000         8.145000
# 588190     39.57300  46.377000         6.804000
# 159725     38.92600  45.162000         6.236000
# 159667     37.45700  43.508000         6.051000
#             ...        ...              ...
# 513090     66.40900  31.546000       -34.863000
# 516100     69.12900  30.691000       -38.438000
# 159851     69.87800  30.913000       -38.965000
# 516860     72.54600  30.841000       -41.705000
# mean       38.76426  31.794009        -6.970258
# 
# [898 rows x 3 columns]
# 2024-10-13 18:19:44.976 | INFO     | backtest.analyze:pipeline:200 - rank_personal: 
#        full_code comparison_experiment          window  bar_num   ann_return  \
# 219932    512690                  base  None-None-None       44  1330201.424   
# 216803    512690                  base  None-None-None       44    81374.850   
# 220017    515220                  base  None-None-None       34    16525.856   
# 219805    159949                  base  None-None-None       48     8103.964   
# 220364    159806                  base  None-None-None       34     7256.258   
#          ...                   ...             ...      ...          ...   
# 220321    159740                  base  None-None-None       20      -95.203   
# 220322    159741                  base  None-None-None       19      -95.639   
# 219964    513130                  base  None-None-None       19      -95.798   
# 219977    513330                  base  None-None-None       23      -96.922   
# 219978    513360                  base  None-None-None       19      -94.567   
# 
#         max_dd       score  
# 219932  42.630  931148.225  
# 216803  42.630   56971.429  
# 220017  22.222   11585.896  
# 219805  36.906    5686.483  
# 220364  37.260    5093.172  
#        ...         ...  
# 220321  61.084     -57.167  
# 220322  60.793     -57.301  
# 219964  61.390     -57.697  
# 219977  61.774     -57.949  
# 219978  65.844     -59.407  
# 
# [18176 rows x 7 columns]
# =============================================================================

# =============================================================================
# 15min
# 2024-10-13 18:27:15.743 | INFO     | backtest.analyze:pipeline:164 - score_mean_base: 32.026
# 2024-10-13 18:27:15.745 | INFO     | backtest.analyze:pipeline:165 - score_mean_strategy_effective: 29.302
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:175: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_rank_df = bacetest_df.groupby('window').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:27:15.764 | INFO     | backtest.analyze:pipeline:176 - strategy_rank: 
# window
# None-None-None    35.403
# 13-28-8           29.875
# 13-27-9           29.875
# 13-28-9           29.875
# 14-26-7           29.875
#  
# 10-27-10          29.242
# 11-24-10          29.202
# 10-24-10          29.182
# 12-25-10          29.155
# 14-24-10          29.137
# Length: 31, dtype: float64
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:179: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   rank_portfolio_df = bacetest_df.groupby('full_code').apply(lambda x: x.score.mean()).sort_values(ascending=False).round(3)
# 2024-10-13 18:27:15.851 | INFO     | backtest.analyze:pipeline:180 - rank_portfolio: 
# full_code
# 159851    32.698
# 513090    32.443
# 159808    32.406
# 516860    32.319
# 516100    32.233
#  
# 515760    26.310
# 517770    25.762
# 159961    25.467
# 159711    25.243
# 159788    24.577
# Length: 911, dtype: float64
# 2024-10-13 18:27:15.853 | INFO     | backtest.analyze:pipeline:192 - IndexError: index 0 is out of bounds for axis 0 with size 0
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:33: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   base_group_df = base_df.groupby('full_code').apply(lambda x: x[column].mean())
# E:\03_software_engineering\github\seagull-quantization\seagull\backtest\analyze.py:34: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#   strategy_group_df = strategy_df.groupby('full_code').apply(lambda x: x[column].mean()) # 取均值，也可以最大值
# 2024-10-13 18:27:16.005 | INFO     | backtest.analyze:pipeline:196 - comparison_portfolio: 
#                 base  strategy  strategy_better
# full_code                                      
# 159502     25.539000  29.34700         3.808000
# 588860     28.055000  31.34100         3.286000
# 561700     26.319000  29.52900         3.210000
# 513300     26.267000  29.38100         3.114000
# 513520     26.570000  29.67800         3.108000
#              ...       ...              ...
# 513090     44.653000  30.13500       -14.518000
# 516100     45.102000  29.30500       -15.797000
# 159851     45.523000  29.36500       -16.158000
# 516860     46.655000  29.44700       -17.208000
# mean       32.025997  29.32443        -2.701569
# 
# [912 rows x 3 columns]
# 2024-10-13 18:27:16.028 | INFO     | backtest.analyze:pipeline:200 - rank_personal: 
#       full_code  comparison_experiment          window  bar_num  ann_return  \
# 698      516860                   base  None-None-None      248      66.124   
# 284      159851                   base  None-None-None      248      62.275   
# 644      516100                   base  None-None-None      248      61.644   
# 515      513090                   base  None-None-None      248      55.740   
# 337      159923                   base  None-None-None      248      43.206   
#         ...                    ...             ...      ...         ...   
# 70999    159711  ablation_macd_freq_15        13-28-10      496      -7.474   
# 24308    517770  ablation_macd_freq_15        11-25-10      496      -7.890   
# 2558     517770  ablation_macd_freq_15        10-24-10      496      -8.071   
# 6245     517770  ablation_macd_freq_15        10-25-10      496      -8.359   
# 20791    517770  ablation_macd_freq_15        11-24-10      496      -8.874   
# 
#        max_dd   score  
# 698    10.327  72.546  
# 284    10.293  69.878  
# 644    11.364  69.129  
# 515     6.393  66.409  
# 337    10.927  56.458  
#       ...     ...  
# 70999  14.662  20.873  
# 24308  15.443  20.318  
# 2558   15.783  20.105  
# 6245   16.337  19.759  
# 20791  17.277  19.168  
# 
# [21568 rows x 7 columns]
# =============================================================================

