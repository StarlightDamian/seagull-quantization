# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:32:33 2024

@author: awei
"""
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import backtrader as bt
import backtrader.strategies as btstrats
import backtrader.analyzers as btanalyzers

from seagull.settings import PATH
from seagull.utils import utils_database



# =============================================================================
# date_start, date_end = '1900-01-01', '2024-05-01'
# with base_connect_database.engine_conn("POSTGRES") as conn:
#     history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{date_start}' AND date < '{date_end}'", con=conn.engine)
# 
# stock_hfq_df = history_day_df[history_day_df.code=='sh.600000']
# stock_hfq_df.to_csv(f'{PATH}/data/stock_hfq_df.csv',index=False)
# #stock_hfq_df = pd.read_excel("./data/sh600000.xlsx",index_col='date',parse_dates=True)
# =============================================================================

#创建一个策略
class TestStrategy(bt.Strategy):
    
    params = (
        ('maperiod', 20),
    )
 #记录功能
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # 引用到close line
        self.dataclose = self.datas[0].close

        # 跟踪订单状态以及买卖价格和佣金
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # 增加移动均线
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态为提交和接受，不做处理
            return

        # 检查订单是否成交
        # 注意，没有足够现金的话，订单会被拒绝。
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # 记录当前处理的close值
        # self.log('Close, %.2f' % self.dataclose[0])

        # 订单是否
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # 大于均线就买
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # 小于均线卖卖卖！
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    #cerebro.broker.setcash(100000.0)

    # 增加一个策略
    cerebro.addstrategy(TestStrategy)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    #获取数据
    stock_hfq_df = pd.read_csv(f'{PATH}/data/stock_hfq_df.csv')
    stock_hfq_df.index=pd.to_datetime(stock_hfq_df.date)
    start_date = datetime(2020, 1, 1)  # 回测开始时间
    end_date = datetime(2024, 1, 1)  # 回测结束时间
    data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
    cerebro.adddata(data)  # 将数据传入回测系统
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    start_cash = 100000.0
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addstrategy(btstrats.SMA_CrossOver)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='SharpeRatio')  # 夏普比率
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='DrawDown')  # 回撤
    cerebro.addanalyzer(btanalyzers.TimeReturn, _name='TimeReturn')  # 不同时段的收益率
    cerebro.addanalyzer(btanalyzers.PyFolio, _name='PyFolio')  # 生成数据兼容pyfolio
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='TradeAnalyzer')  # 交易统计信息，如获胜、失败次数
    cerebro.addanalyzer(btanalyzers.SharpeRatio_A, _name='SharpeRatio_A')  # 年化夏普比率
    cerebro.addanalyzer(btanalyzers.Transactions, _name='Transactions')#  每笔交易的标的、价格、数量等信息
    

    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    thestrats = cerebro.run()
    
    port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    pnl = port_value - start_cash  # 盈亏统计
    print(f"初始资金:{start_cash}\n回测期间:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}")
    print(f"总资金:{round(port_value, 2)}")
    print(f"净收益:{round(pnl, 2)}")
    cerebro.plot(style='candlestick')
    
    thestrat = thestrats[0]
    print('Sharpe Ratio:',thestrat.analyzers.SharpeRatio.get_analysis())
    print('Annual Return:', thestrat.analyzers.AnnualReturn.get_analysis())
    print('Drawdown:', thestrat.analyzers.DrawDown.get_analysis())
    print('SharpeRatio_A:', thestrat.analyzers.SharpeRatio_A.get_analysis())
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    plt.figure()
    cerebro.plot()
    
    
    transactions_dict = thestrat.analyzers.Transactions.get_analysis()
    transactions_df = pd.DataFrame(list(transactions_dict.items()), columns=['date', 'Column_Values']) \
    .assign(amount=lambda df: df['Column_Values'].apply(lambda x: x[0][0]),
            price=lambda df: df['Column_Values'].apply(lambda x: x[0][1]),
            sid=lambda df: df['Column_Values'].apply(lambda x: x[0][2]),
            symbol=lambda df: df['Column_Values'].apply(lambda x: x[0][3]),
            value=lambda df: df['Column_Values'].apply(lambda x: x[0][4])) \
    .drop('Column_Values', axis=1)
    
    #amount: 交易数量
    #price: 交易价格
    #sid: 交易 ID 或标识符
    #symbol: 交易标的物的符号或名称
    #value: 交易总价值或净额
    
    pyfoliozer = thestrat.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()



# =============================================================================
#     import pyfolio as pf
#     pf.create_full_tear_sheet(
#         returns,
#         positions=positions,
#         transactions=transactions,
#         #gross_lev=gross_lev,
#         live_start_date='2020-02-21',  # This date is sample specific
#         round_trips=True)
# =============================================================================
    import quantstats as qs
#import quantstats
    df = self.feed.get_df('000300.SH')
    df= df[['rate']]
    df.index = pd.to_datetime(df.index)
    print(df)
    qs.reports.html(returns,
                    benchmark=returns,#bench_returns
                    output='stats.html',
                    title='1')#'stock:'+stock_symbol+' bench:'+bench_symbol
    plt.figure()
    qs.plots.snapshot(returns, title='Facebook Performance', show=True)
# =============================================================================
#     TimeDrawDown： 指定时间粒度下的回撤GrossLeverage： 总杠杆PositionsValue： 持仓价值PyFolio： 生成数据兼容pyfolioLogReturnsRolling： 滚动log收益率PeriodStats： 给定时段下的基本统计信息Returns： 用对数法计算总、平均、复合、年化收益率SharpeRatio： 夏普比率SharpeRatio_A： 年化夏普比率SQN：系统质量数System Quality NumberTimeReturn：不同时段的收益率TradeAnalyzer：交易统计信息，如获胜、失败次数等Transactions： 每笔交易的标的、价格、数量等信息VWR： variability weighted return，波动率加权的收益率常用的Analyzer分析器有：AnnualReturn， DrawDown， PyFolio， SharpeRatio， TimeReturn。
# =============================================================================

