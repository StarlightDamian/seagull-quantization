# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:46:22 2024

@author: awei
回测基类(backtest_base)
不能开VPN
"""

from datetime import datetime
import backtrader as bt
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import efinance
from collections import defaultdict


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


class BaseStrategy(bt.Strategy):
    def __init__(self):
        # 记录买入和卖出的订单
        self.buy_bond_record = defaultdict(lambda: defaultdict(list))
        self.sell_bond_record = defaultdict(lambda: defaultdict(list))

    def log_order(self, order, action):
        """记录订单信息"""
        today_time_string = self.datetime.datetime().strftime('%Y-%m-%d')
        record = {
            "order_ref": order.ref,
            "bond_name": order.data._name,
            "size": order.size,
            "price": order.executed.price,
            "value": order.executed.value,
            "trade_date": self.datetime.datetime(0),
            "commission": order.executed.comm
        }

        if action == "buy":
            self.buy_bond_record[today_time_string][order.data._name.replace(".", "_")].append(record)
            logger.debug(f'{self.datetime.date()} 订单{order.ref} 已购入 {order.data._name} , 购入单价 {order.executed.price:.2f}, 数量 {order.size}, 费用 {order.executed.value:.2f}, 手续费 {order.executed.comm:.2f}')
        elif action == "sell":
            record["value"] = -order.executed.price * order.size
            record["sell_type"] = order.info.sell_type if hasattr(order.info, 'sell_type') else None
            self.sell_bond_record[today_time_string][order.data._name.replace(".", "_")].append(record)
            logger.debug(f'{self.datetime.date()} 订单{order.ref} 已卖出 {order.data._name}, 卖出金额 {order.executed.price:.2f}, 数量 {order.size}, 费用 {-order.executed.price * order.size:.2f}, 手续费 {order.executed.comm:.2f}')

    def notify_order(self, order):
        """通知订单状态,当订单状态变化时触发"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log_order(order, "buy")
            elif order.issell():
                self.log_order(order, "sell")
        elif order.status == order.Canceled:
            logger.debug(f"{self.datetime.date()} 订单{order.ref} 已取消")
        elif order.status == order.Margin:
            logger.warning(f'{self.datetime.date()} 订单{order.ref} 现金不足、金额不足拒绝交易')
        elif order.status == order.Rejected:
            logger.warning(f'{self.datetime.date()} 订单{order.ref} 拒绝交易')
        elif order.status == order.Expired:
            logger.warning(f'{self.datetime.date()} 订单{order.ref} 超过有效期已取消, 订单开价 {order.price}, 当天最高价{order.data.high[0]}, 最低价{order.data.low[0]}')

    def log(self, txt, dt=None):
        """日志记录功能，可选时间戳"""
        dt = dt or self.datetime.date(0)
        logger.info(f'{dt.isoformat()} - {txt}')

    def stop(self):
        """策略结束时调用，可以用于输出最终结果"""
        logger.info(f'策略结束: 资金 {self.broker.getvalue():.2f}')


class MyStrategy1(BaseStrategy):
    def __init__(self):
        super(MyStrategy1, self).__init__()
        self.close_price = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=5)

    def next(self):
        if self.close_price[0] > self.sma[0]:
            self.log(f"buy 500 in {self.datetime.date()}, 预期购入金额 {self.data.close[0]}, 剩余可用资金 {self.broker.getcash()}")
            self.buy(size=500, price=self.data.close[0])
        if self.position and self.close_price[0] < self.sma[0]:
            self.log(f"sell in {self.datetime.date()}, 预期卖出金额 {self.data.close[0]}, 剩余可用资金 {self.broker.getcash()}")
            self.sell(size=500, price=self.data.close[0])


if __name__ == '__main__':
    # 获取数据
    start_time = datetime(2015, 1, 1)
    end_time = datetime(2021, 1, 1)
    dataframe = get_k_data('600519', begin=start_time, end=end_time)
    # =============== 为系统注入数据 =================
    # 加载数据
    data = bt.feeds.PandasData(dataname=dataframe, fromdate=start_time, todate=end_time)
    # 初始化cerebro回测系统
    cerebro_system = bt.Cerebro()  # Cerebro引擎在后台创建了broker(经纪人)实例，系统默认每个broker的初始资金量为10000
    # 将数据传入回测系统
    cerebro_system.adddata(data)  # 导入数据，在策略中使用 self.datas 来获取数据源
    # 将交易策略加载到回测系统中
    cerebro_system.addstrategy(MyStrategy1)
    # =============== 系统设置 ==================
    # 设置启动资金为 1000000
    start_cash = 1000000
    cerebro_system.broker.setcash(start_cash)
    # 设置手续费 万2.5
    cerebro_system.broker.setcommission(commission=0.00025)
    logger.debug(f'初始资金: {start_cash} 回测期间：from {start_time} to {end_time}')
    # 运行回测系统
    cerebro_system.run()
    # 获取回测结束后的总资金
    portvalue = cerebro_system.broker.getvalue()
    pnl = portvalue - start_cash
    # 打印结果
    logger.info(f'净收益: {pnl}')
    logger.info(f"总资金: {portvalue}")
    cerebro_system.plot(style='candlestick')
    plt.show()
