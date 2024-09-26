# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:46:22 2024

@author: awei
backtrader_multiprocessing
"""

from datetime import datetime
import backtrader as bt
import logging
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import efinance
import multiprocessing as mp


def get_k_data(stock_code, begin: datetime, end: datetime) -> pd.DataFrame:
    """
    根据efinance工具包获取股票数据
    :param stock_code:股票代码
    :param begin: 开始日期
    :param end: 结束日期
    :return:
    """
    k_dataframe: pd.DataFrame = efinance.stock.get_quote_history(
        stock_code, beg=begin.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    k_dataframe = k_dataframe.iloc[:, :9]
    k_dataframe.columns = ['name', 'code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover']
    k_dataframe.index = pd.to_datetime(k_dataframe.date)
    k_dataframe.drop(['name', 'code', 'date'], axis=1, inplace=True)
    return k_dataframe


class MyStrategy(bt.Strategy):  # 策略
    def __init__(self):
        self.close_price = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=5)

    def notify_order(self, order):  # 固定写法，查看订单情况
        if order.status in [order.Submitted, order.Accepted]:  # 接受订单交易，正常情况
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info('已买入, 购入金额 %.2f' % order.executed.price)
            elif order.issell():
                logger.info('已卖出, 卖出金额 %.2f' % order.executed.price)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.info('订单取消、保证金不足、金额不足拒绝交易')

    def next(self):
        if self.close_price[0] > self.sma[0]:
            logger.info("buy 500 in {}, 预期购入金额 {}, 剩余可用资金 {}", self.datetime.date(), self.data.close[0], self.broker.getcash())
            self.buy(size=500, price=self.data.close[0])
        if self.position:
            if self.close_price[0] < self.sma[0]:
                logger.info("sell in {}, 预期卖出金额 {}, 剩余可用资金 {}", self.datetime.date(), self.data.close[0], self.broker.getcash())
                self.sell(size=500, price=self.data.close[0])


def run_backtest(stock_code):
    start_time = datetime(2015, 1, 1)
    end_time = datetime(2021, 1, 1)

    dataframe = get_k_data(stock_code, begin=start_time, end=end_time)
    data = bt.feeds.PandasData(dataname=dataframe, fromdate=start_time, todate=end_time, name=stock_code)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy)

    start_cash = 1000000
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.00025)
    
    logger.debug('初始资金: {} 回测期间：from {} to {}'.format(start_cash, start_time, end_time))
    cerebro.run()

    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - start_cash
    
    logger.info('股票代码: {}, 净收益: {}, 总资金: {}'.format(stock_code, pnl, portvalue))
    cerebro.plot(style='candlestick')
    plt.show()


if __name__ == '__main__':
    stock_codes = ['600519', '000001', '000002', '000003']  # 示例股票代码，可替换成实际的股票代码列表

    # 使用多进程进行回测
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(run_backtest, stock_codes)
