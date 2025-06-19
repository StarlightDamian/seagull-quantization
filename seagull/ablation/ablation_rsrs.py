# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:42:46 2025

@author: awei
ablation_rsrs
"""
import os
import itertools

import vectorbt as vbt
import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_log
from backtest import vectorbt_macd, analyze

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

# 定义RSRS计算函数
def calculate_rsrs_matrix(highs, lows, window=18):
    """ 向量化RSRS计算 """
    # 为每个日期计算RSRS：我们用最低价和最高价的对数变化来计算RSRS
    rsrs_matrix = np.full((len(highs), len(highs.columns)), np.nan)

    for i in range(window, len(highs)):
        #print(i)
        x = lows.iloc[i - window:i].values  # 低点作为自变量X
        y = highs.iloc[i - window:i].values  # 高点作为因变量Y
        #slopes = np.polyfit(x.T, y.T, 1)[0]  # 对所有股票计算线性回归的斜率
        slopes = np.polyfit(x.flatten(), y.flatten(), 1)

        rsrs_matrix[i] = slopes

    return pd.DataFrame(rsrs_matrix, index=highs.index, columns=highs.columns)

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
# import vectorbt as vbt
# import numpy as np
# import pandas as pd
# from seagull.settings import PATH
# 
# symbols = ["ADA-USD", "ETH-USD"]
# 
# # 下载历史数据（包含高、低、收盘价）
# data = vbt.YFData.download(
#     symbols,
#     start='2020-01-01',
#     end='2023-01-01'
# )
# highs = data.get('High')  # 最高价序列
# lows = data.get('Low')    # 最低价序列
# closes = data.get('Close') # 收盘价序列
# 
# # 定义RSRS计算函数
# def calculate_rsrs_matrix(highs, lows, window=18):
#     """ 向量化RSRS计算 """
#     # 为每个日期计算RSRS：我们用最低价和最高价的对数变化来计算RSRS
#     rsrs_matrix = np.full((len(highs), len(highs.columns)), np.nan)
# 
#     for i in range(window, len(highs)):
#         #print(i)
#         x = lows.iloc[i - window:i].values  # 低点作为自变量X
#         y = highs.iloc[i - window:i].values  # 高点作为因变量Y
#         #slopes = np.polyfit(x.T, y.T, 1)[0]  # 对所有股票计算线性回归的斜率
#         slopes = np.polyfit(x.flatten(), y.flatten(), 1)
# 
#         rsrs_matrix[i] = slopes
# 
#     return pd.DataFrame(rsrs_matrix, index=highs.index, columns=highs.columns)
# 
# # 计算所有股票的RSRS
# rsrs = calculate_rsrs_matrix(highs, lows, window=18)
# 
# # 标准化RSRS（Z-Score）
# zscore_window = 252  # 一年的交易日
# rsrs_mean = rsrs.rolling(zscore_window).mean()
# rsrs_std = rsrs.rolling(zscore_window).std()
# rsrs_z = (rsrs - rsrs_mean) / rsrs_std
# 
# # 定义买卖信号（Z > 1买入，Z < -1卖出）
# entries = rsrs_z > 1   # 买入信号
# exits = rsrs_z < -1    # 卖出信号
# 
# # 构建投资组合
# portfolio = vbt.Portfolio.from_signals(
#     close=closes,      # 收盘价作为交易价格
#     entries=entries,    # 买入信号
#     exits=exits,        # 卖出信号
#     fees=0.001,         # 交易费率0.1%
#     freq='1D'           # 日频数据
# )
# 
# # 输出绩效报告
# print(portfolio.stats())
# 
# # 遍历symbols，分别保存每只股票的图表
# for symbol in symbols:
#     # 获取当前股票的 portfolio
#     stock_portfolio = portfolio[symbol]
# 
#     # 生成图表并保存为HTML
#     fig = stock_portfolio.plot()  # 生成图表
#     fig.write_html(f"{PATH}/seagull/html/{symbol}.html") 
# =============================================================================
