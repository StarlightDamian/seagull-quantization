# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:19:16 2023

@author: awei
"""

class StockTradingStrategy:
    def __init__(self, initial_balance=100000, buy_threshold=0.05, sell_threshold=0.10):
        self.balance = initial_balance
        self.stock_holdings = 0
        self.buy_threshold = buy_threshold  # 股价上涨5%时加仓
        self.sell_threshold = sell_threshold  # 股价下跌10%时清仓

    def execute_trade(self, current_price):
        if current_price >= (1 + self.buy_threshold) * self.buy_threshold:
            # 股价上涨5%，加仓5股
            additional_shares = 5
            cost = additional_shares * current_price
            if cost <= self.balance:
                self.stock_holdings += additional_shares
                self.balance -= cost
                print(f"买入 {additional_shares} 股，当前余额: {self.balance}")
            else:
                print("余额不足，无法购买更多股票")
        elif current_price <= (1 - self.sell_threshold) * self.buy_threshold:
            # 股价下跌10%，清仓
            sale_proceeds = self.stock_holdings * current_price
            self.balance += sale_proceeds
            self.stock_holdings = 0
            print(f"清仓，卖出 {self.stock_holdings} 股，当前余额: {self.balance}")
        else:
            print("未执行交易")

# 使用示例
if __name__ == "__main__":
    strategy = StockTradingStrategy(initial_balance=100000, buy_threshold=0.05, sell_threshold=0.10)
    
    # 模拟股价波动
    stock_prices = [100, 105, 110, 115, 105, 120, 130, 140, 135, 125]
    
    for price in stock_prices:
        strategy.execute_trade(price)
