# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:34:54 2024

@author: awei
ETF板块(data_api_adata_etf)
"""
import adata

def api_adata_etf():
    etf_df = adata.fund.info.all_etf_exchange_traded_info()
    return etf_df

# 获取ETF的行情信息-日、周、月 k线	，同花顺，同一只股的，日级
# ['fund_code', 'trade_time', 'trade_date', 'open', 'high', 'low', 'close','volume', 'amount', 'change', 'change_pct']	(1943, 11)
etf_df = adata.fund.market.get_market_etf() 

#获取ETF的行情-当日分时，同花顺，价格是开盘价，同一只股的，分钟级
#['fund_code', 'trade_time', 'trade_date', 'price', 'avg_price', 'volume','amount', 'change', 'change_pct'] (242, 9)
etf_min_df = adata.fund.market.get_market_etf_min()	

# 获取当前单一股行情（包含个股、ETF），同花顺，优先个股
#['fund_code', 'trade_time', 'trade_date', 'open', 'high', 'low', 'price','volume', 'amount', 'change', 'change_pct'] (1, 11)
etf_current_df = adata.fund.market.get_market_etf_current(fund_code='603062')	

# 获取所有A股市场的ETF信息，东方财富（场内）
# ['fund_code', 'short_name', 'net_value']	(951, 3)
etf_exchange_df = adata.fund.info.all_etf_exchange_traded_info()


if __name__ == '__main__':
    import efinance as ef # efinance不能连国际VPN 
    stock_code = ['513100','159660','513520','159659','430489'] 
    frequency = 5 # frequency in [5, 15, 30, 60] 
    df = ef.stock.get_quote_history(stock_code, klt=frequency) # 优先个股，其次ETF
    print(df)

