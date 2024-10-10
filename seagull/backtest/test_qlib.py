# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 00:00:47 2024

@author: awei
"""

import qlib  
from qlib.contrib.model.stock_rnn import LSTNet  
from qlib.contrib.data.handler import Alpha158  
from qlib.contrib.strategy.strategy import TopkDropoutStrategy  
from qlib.contrib.backtest.backtest import BacktestEngine, default_setup  
from qlib.contrib.evaluate.backtest_metric import calculate_backtest_metrics  
from qlib.utils import init_env  

# 初始化环境  
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")  
  
# 加载数据处理器  
handler = Alpha158(qlib.config.get_instance().qlib_data_dir, stock_list=["000001.SZ"])  # 以平安银行为例  
  
# 定义模型  
model = LSTNet(handler)  
  
# 训练模型（这里省略了训练过程，通常你需要使用历史数据来训练模型）  
# model.fit(...)  
  
# 定义策略  
strategy = TopkDropoutStrategy(model, handler, topk=10, n_drop=5)  
  
# 设置回测环境  
engine = BacktestEngine()  
engine.set_backtest_params(  
    initial_cash=1000000,  # 初始资金  
    benchmark="000300.SH",  # 基准指数  
    start_time="2020-01-01",  # 回测开始时间  
    end_time="2023-01-01",  # 回测结束时间  
    frequency="day",  # 交易频率  
    trade_calendar=qlib.config.get_instance().trade_calendar,  # 交易日历  
)  
  
# 执行回测  
records = engine.backtest(strategy)  
  
# 计算回测指标  
metrics = calculate_backtest_metrics(records, benchmark="000300.SH")  
print(metrics)  
  
# 可视化回测结果（需要安装 matplotlib）  
import matplotlib.pyplot as plt  

# 绘制收益曲线  
plt.figure(figsize=(14, 7))  
engine.plot_curve(records)  
plt.show()  
  
# 绘制持仓情况  
plt.figure(figsize=(14, 7))  
engine.plot_positions(records)  
plt.show()