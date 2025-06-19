# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:55:56 2025

@author: awei
(ablation_rsi_benchmark)
"""
import os

import vectorbt as vbt
import pandas as pd
import talib
# import numpy as np
# import matplotlib.pyplot as plt

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

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

if __name__ == '__main__':
    # data = vbt.YFData.download('AAPL', start='2018-01-01', end='2022-12-31')
    # price = data.get('Close')  # 获取收盘价
    full_code_list = ['588310.sh',
                    '159707.sz',
                    '515760.sh',
                    '510530.sh',
                    '516950.sh',
                    '515900.sh',
                    '512960.sh',
                    '159959.sz',
                    '515150.sh',
                    ]
    full_code_list = ['159895.sz']
    for full_code in full_code_list:
        print(full_code)
        with utils_database.engine_conn("POSTGRES") as conn:
            raw_df = pd.read_sql(f"select primary_key, date,high, low, close from dwd_ohlc_incr_stock_daily where full_code='{full_code}'", con=conn.engine)
        
        raw_df = raw_df.drop_duplicates('primary_key', keep='first')
        raw_df.sort_values(by='date', ascending=True, inplace=True)
        
        # raw_df = raw_df[raw_df.date<'2024-10-08']
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        price = raw_df.set_index('date')['close']
        
        # 重采样为周线数据（'W' 是表示按周重采样的频率）
        #price = price.resample('W').last()  # 使用每周最后一个交易日的收盘价
        
        # 或者重采样为月线数据（'M' 是表示按月重采样的频率）
        #price = price.resample('M').last()  # 使用每月最后一个交易日的收盘价
        
        # 2. 定义一个简单的策略
        #fast_ma = price.vbt.rolling_mean(window=20)  # 20日均线
        #slow_ma = price.vbt.rolling_mean(window=50)  # 50日均线
        macd = vbt.MACD.run(
            close=price,  # close: 2D数组，表示收盘价
            fast_window=7,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
            slow_window=24,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
            signal_window=12,  # 信号线的窗口大小,Signal line period, default value 9
    # =============================================================================
    #         fast_window=12,  # 快速移动平均线的窗口大小,Fast EMA period, default value 12
    #         slow_window=26,  # 慢速移动平均线的窗口大小,Slow EMA period, default value 26
    #         signal_window=9,  # 信号线的窗口大小,Signal line period, default value 9
    # =============================================================================
            macd_ewm=False,  # 布尔值，是否使用指数加权移动平均（EMA）计算MACD线，True:EMA, False:SMA
            signal_ewm=True,  #布尔值，是否使用EMA计算信号线，True:EMA, False:SMA
            adjust=False,  #布尔值，是否在计算EMA时进行调整
            # cache_dict, 字典，用于缓存计算结果
        )
        # 买入信号：当快线突破慢线时
        # entries = fast_ma < slow_ma
        
        # 卖出信号：当快线跌破慢线时
        # exits = fast_ma > slow_ma
        
    # =============================================================================
    #     9
    #     24
    #     12
    #     
    #     6-7
    #     24
    #     12
    # =============================================================================
        
        # 计算 DIF 和 DEA 的斜率
        dif_slope = macd.macd.diff()  # DIF 线的斜率
        #dea_slope = macd.signal.diff()  # DEA 线的斜率
        
        # 买入信号：DIF 的斜率从负变正，且 DEA 的斜率也从负变正
        #entries = (dif_slope > 0) & (dif_slope.shift(1) <= 0)  # & (dea_slope > 0) & (dea_slope.shift(1) < 0)
        
        # 卖出信号：DIF 的斜率从正变负
        #exits = (dif_slope <= 0) & (dif_slope.shift(1) > 0)
        #raw_df['adx'] = talib.ADX(raw_df.high, raw_df.low, raw_df.close, timeperiod=14)
        rsi = vbt.RSI.run(price, window=14).rsi
        #lower, upper = winsorize(rsi, window=40, n_lower=2.5, n_upper=2.5)
        
        # 买入信号：当前RSI小于窗口内的标准差
        #entries = rsi < lower  # RSI小于标准差，视为买入信号
        #adx = raw_df.set_index('date')['adx']
        # 卖出信号：当前RSI大于窗口内的标准差
        #exits = rsi > upper  # RSI大于标准差，视为卖出信号
        #entries = (rsi < 30) &(adx<20) # 买入信号：RSI < 30
        #exits = (rsi > 70)&(adx<20)  # 卖出信号：RSI > 70
        entries = (rsi < 15)
        #exits = (rsi > 40)
       # exits = (dif_slope <= 0) & (dif_slope.shift(1) > 0)
        #exits = macd.macd < macd.signal  # DIF < DEA (死叉)
        exits = (macd.macd.shift(1) > macd.signal.shift(1)) & (macd.macd < macd.signal)  # 只在快线从上穿越慢线时卖出
    
    # =============================================================================
    # 金叉：Golden Cross
    # 死叉：Death Cross
    #     # 买入信号：MACD 线从下方突破信号线（金叉）
    #     entries = macd.macd > macd.signal  # DIF > DEA (金叉)
    #     
    #     # 卖出信号：MACD 线从上方跌破信号线（死叉）
    #     exits = macd.macd < macd.signal  # DIF < DEA (死叉)
    # =============================================================================
        
        # 可选：若要在信号发生时触发，可以使用 .shift() 来避免未来数据泄露
        # entries = (entries) & (entries.shift(1) == False)  # 只在从无信号到有信号时触发买入
        # exits = (exits) & (exits.shift(1) == False)  # 只在从无信号到有信号时触发卖出
        
        # 3. 运行回测
        portfolio = vbt.Portfolio.from_signals(price,
                                               entries,
                                               exits,
                                               freq='d',
                                               init_cash=10000,
                                               fees=0.001,
                                               slippage=0.001,
                                               #size=100,
                                               )
        orders = portfolio.orders.records_readable
        
        orders.to_csv(f'{PATH}/data/orders_records_readable2.csv', index=False)
        
        
        # 4. 计算相关的绩效指标
        # orders = portfolio.orders
        # trades_records_readable
        
        # 5. 输出回测结果到 HTML 文件
        # portfolio.stats.to_html(f"{PATH}/seagull/html/backtest_report.html")
        fig = portfolio.plot()  # .figure
        fig.write_html(f"{PATH}/html/{full_code}.html")
        logger.info(portfolio.stats(settings=dict(freq='d',
                                                  year_freq='243 days')))
        
        logger.info(portfolio.returns_stats(settings=dict(freq='d'),
                                            year_freq='243 days',))
        
        # 每月收益
        daily_returns = portfolio.returns()
        # 按月聚合（计算每个月的累计收益）
        monthly_returns = daily_returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        # 转换为百分比并保留两位小数
        monthly_returns_percent = monthly_returns.round(4) * 100  # 转换为百分比
        
        std_returns = monthly_returns.std()
    
    
        stats_metrics_dict={"Start": "start",
                                    "End": "end",
                                    "Period": "period",
                                    "Start Value": "start_value",
                                    "End Value": "end_value",
                                    "Total Return [%]": "total_return",
                                    "Benchmark Return [%]": "benchmark_return",
                                    "Max Gross Exposure [%]": "max_gross_exposure",
                                    "Total Fees Paid": "total_fees_paid",
                                    "Max Drawdown [%]": "max_dd",
                                    "Max Drawdown Duration": "max_dd_duration",
                                    "Total Trades": "total_trades",
                                    "Total Closed Trades": "total_closed_trades",
                                    "Total Open Trades": "total_open_trades",
                                    "Open Trade PnL": "open_trade_pnl",
                                    "Win Rate [%]": "win_rate",
                                    "Best Trade [%]": "best_trade",
                                    "Worst Trade [%]": "worst_trade",
                                    "Avg Winning Trade [%]": "avg_winning_trade",
                                    "Avg Losing Trade [%]": "avg_losing_trade",
                                    "Avg Winning Trade Duration": "avg_winning_trade_duration",
                                    "Avg Losing Trade Duration": "avg_losing_trade_duration",
                                    "Profit Factor": "profit_factor",
                                    "Expectancy": "expectancy",
                                    "Sharpe Ratio": "sharpe_ratio",
                                    "Calmar Ratio": "calmar_ratio",
                                    "Omega Ratio": "omega_ratio",
                                    "Sortino Ratio": "sortino_ratio",
                                    }
        
        returns_stats_metrics_dict={#'Start': "start",
                                    #'End': "end",
                                    #'Period': "period",
                                    #'Total Return [%]': "total_return",
                                    #'Benchmark Return [%]': "benchmark_return",
                                    "Annualized Return [%]": "ann_return",
                                    "Annualized Volatility [%]": "ann_volatility",
                                    #'Max Drawdown [%]': "max_dd",
                                    #'Max Drawdown Duration': "max_dd_duration",
                                    #'Sharpe Ratio': "sharpe_ratio",
                                    #'Calmar Ratio': "calmar_ratio",
                                    #'Omega Ratio': "omega_ratio",
                                    #'Sortino Ratio': "sortino_ratio",
                                    'Skew': 'skew',
                                    'Kurtosis': 'kurtosis',
                                    'Tail Ratio': 'tail_ratio',
                                    'Common Sense Ratio': 'common_sense_ratio',
                                    'Value at Risk': 'value_at_risk',
                                    'Alpha': 'alpha',
                                    'Beta': 'beta'}
# =============================================================================
#     backtest_df = portfolio.returns_stats(metrics=self.returns_stats_metrics_dict.values(),
#                                                   settings=dict(freq=freq),
#                                                   year_freq='243 days',
#                                                   agg_func=None)
#     backtest_df = portfolio.stats(
#                             metrics=self.stats_metrics_dict.values(),  # ['sharpe_ratio', 'max_dd']
#                             settings=dict(freq=freq,
#                                           year_freq='243 days'),  # freq in ['d','30d','365d']
#                             #group_by=False,
#                             agg_func=None)
# =============================================================================
# =============================================================================
# import vectorbt as vbt
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# 
# # 生成回测报告的函数
# def generate_backtest_report(price_data, rsi_window=14, rsi_entry=30, rsi_exit=70):
#     # 计算RSI指标
#     rsi = vbt.RSI.run(price_data, window=rsi_window).rsi
#     
#     # 生成进场和出场信号
#     entries = rsi < rsi_entry
#     exits = rsi > rsi_exit
#     
#     # 运行回测
#     cumulative_returns = portfolio.cumulative_returns()
# =============================================================================
