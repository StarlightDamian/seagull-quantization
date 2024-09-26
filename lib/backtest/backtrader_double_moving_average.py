# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:29:20 2024

@author: awei
(backtrader_double_moving_average)
"""

import backtrader as bt

class DualMovingAverageStrategy(bt.Strategy):
    params = (
        ('short_period', 50),  # 短线周期
        ('long_period', 200),  # 长线周期
    )

    def __init__(self):
        # 添加短期和长期均线指标
        self.short_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.long_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)

    def next(self):
        if self.short_sma > self.long_sma:
            if not self.position:  # 如果没有持仓
                self.buy()  # 买入信号
        elif self.short_sma < self.long_sma:
            if self.position:  # 如果有持仓
                self.sell()  # 卖出信号

# 创建Cerebro引擎
cerebro = bt.Cerebro()

# 添加数据源
data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2020, 1, 1), todate=datetime(2021, 1, 1))
cerebro.adddata(data)

# 添加策略
cerebro.addstrategy(DualMovingAverageStrategy)

# 设置初始资金
cerebro.broker.setcash(10000.0)

# 设置固定的佣金
cerebro.broker.setcommission(commission=0.001)

# 运行策略
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 绘制结果
cerebro.plot()
