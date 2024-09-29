# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:49:48 2023

@author: awei
手续费
handling fee
A股交易手续费分为三笔，佣金、印花税和过户费，后两笔为国家统一征收:
1.佣金，需转接人工客服验证账户后查询确认
双向收取，单笔5元起收:
2.印花税，单向收取，卖出时收取成交金额的1%o;
3.过户费，双向收取，上海股票收，深圳不收为股票成交金额的0.01%。。

证券公司手续费计算:
1、证监会规定:交易佣金不超过千分之3，交易佣金不足5元按5元收取2、手续费构成:交易手续费=净佣金+规费+过户费+印花税(单边)
(1)净佣金:即券商收取的费用，成本在万分之1左右。
(2)规费: 即证管费和经手费的统称，证管费万分之0.2，经手费万分之0.487，由券商代交易所收取
(3)过户费:即交易所收取的股票过户的费用，万分之0.2，目前大部分券商只收上交所的过户费.
(4) 印花税:国家财政税收，仅卖出单边收取
3.券商佣金计算方式:一般说的金为净佣金+规费
"""
class TradingSimulation:
    def __init__(self, initial_balance=1000000):
        self.balance = initial_balance
        self.transaction_history = []

    def execute_trade(self, symbol, amount, price, is_buy=True):
        if is_buy:
            cost = amount * price
            if cost > self.balance:
                raise ValueError("Insufficient balance to execute the buy order.")
            self.balance -= cost
        else:
            revenue = amount * price
            self.balance += revenue
        self.transaction_history.append({
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "is_buy": is_buy
        })

    def get_balance(self):
        return self.balance

    def get_transaction_history(self):
        return self.transaction_history

# 使用示例
if __name__ == "__main__":
    trading = TradingSimulation(initial_balance=1000000)
    trading.execute_trade("AAPL", 100, 150.0, is_buy=True)
    trading.execute_trade("AAPL", 50, 155.0, is_buy=True)
    trading.execute_trade("AAPL", 30, 160.0, is_buy=False)

    print("当前余额:", trading.get_balance())
    print("交易历史:")
    for transaction in trading.get_transaction_history():
        print(transaction)


# =============================================================================
# # 定义交易参数
# transaction_amount = 10000  # 假设交易金额为10,000元
# buying_price = 10.00  # 假设买入价格为10元/股
# selling_price = 11.00  # 假设卖出价格为11元/股
# 
# # 计算佣金
# commission_rate = 0.003  # 佣金率为千分之3
# minimum_commission = 5  # 单笔佣金5元起收
# commission = max(transaction_amount * commission_rate, minimum_commission)  # 佣金 = max(交易金额 * 佣金率, 最低佣金)
# 
# # 计算印花税（卖出时收取）
# stamp_duty_rate = 0.001  # 印花税率为千分之1
# selling_stamp_duty = transaction_amount * stamp_duty_rate  # 卖出印花税 = 交易金额 * 印花税率
# 
# # 计算过户费
# transfer_fee_rate = 0.0001  # 过户费率为万分之1
# transfer_fee = transaction_amount * transfer_fee_rate  # 过户费 = 交易金额 * 过户费率
# 
# # 计算净佣金（不包括规费、印花税、过户费）
# net_commission = commission - (selling_stamp_duty + transfer_fee)  # 净佣金 = 佣金 - (卖出印花税 + 过户费)
# 
# # 计算规费
# regulatory_fee_rate = 0.00002  # 证管费率为万分之0.2
# handling_fee_rate = 0.0000487  # 经手费率为万分之0.487
# regulatory_fees = (regulatory_fee_rate + handling_fee_rate) * transaction_amount  # 规费 = (证管费率 + 经手费率) * 交易金额
# 
# # 计算交易手续费
# transaction_fees = net_commission + regulatory_fees + transfer_fee + selling_stamp_duty  # 交易手续费 = 净佣金 + 规费 + 过户费 + 卖出印花税
# 
# # Output results
# print("Commission:", commission, "yuan")
# print("Selling Stamp Duty:", selling_stamp_duty, "yuan")
# print("Transfer Fee:", transfer_fee, "yuan")
# print("Net Commission:", net_commission, "yuan")
# print("Regulatory Fees:", regulatory_fees, "yuan")
# print("Transaction Fees:", transaction_fees, "yuan")
# 
# # =============================================================================
# # print("交易佣金:", 佣金, "元")
# # print("卖出印花税:", 卖出印花税, "元")
# # print("过户费:", 过户费, "元")
# # print("净佣金:", 净佣金, "元")
# # print("规费:", 规费, "元")
# # print("交易手续费:", 交易手续费, "元")
# # =============================================================================
# =============================================================================
