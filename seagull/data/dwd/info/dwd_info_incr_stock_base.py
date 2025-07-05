# -*- coding: utf-8 -*-
"""
@Date: 2024/5/22 22:23
@Author: Damian
@Email: zengyuwei1995@163.com
@File: full_trading_day.py
@Description: 上市公司信息(data_dwd_info_incr_stock_base)
更新周期：7天
问题记录：
1.ST是带帽和摘帽
2.股票发行日期。新股，次新股
3.北交所是交易所，概念和‘主板’，‘创业板’其实不一样
4.ETF科创板也是20cm

exchange：表示交易所，可以存储交易所的名称，如"北交所"。
"""
import pandas as pd

from seagull.utils import utils_data, utils_database
from seagull.utils.api import utils_api_baostock


def dwd_stock_base():
    # ods_info_stock_incr_baostock.columns = ['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status',
    # 'insert_timestamp']
    # ods_info_stock_incr_baostock_trade.columns = ['code', 'tradeStatus', 'code_name', 'date', 'insert_timestamp']
    with utils_database.engine_conn("POSTGRES") as conn:
        sql = """
        SELECT 
            *
        FROM
            ods_info_stock_incr_baostock_trade 
        WHERE
            date = (SELECT MAX(date) FROM ods_info_stock_incr_baostock_trade)
        """
        trade_df = pd.read_sql(sql, con=conn.engine)
        stock_df = pd.read_sql("ods_info_stock_incr_baostock", con=conn.engine)
        asset_df = pd.read_sql("ods_base_full_asset_details", con=conn.engine)
    stock_df["board_type"] = stock_df["type"].map({"1": "股票",
                                                   "2": "指数",
                                                   "3": "其它",
                                                   "4": "可转债",
                                                   "5": "ETF"})

    trade_df[["board_type", "price_limit_rate"]] = ''
    trade_df.loc[trade_df.code.str.contains("sh.60|sz.00"), 'board_type'] = '主板'
    trade_df.loc[trade_df.code.str.contains(".300|sz.301"), 'board_type'] = '创业板'
    trade_df.loc[trade_df.code.str.contains(".688|.689"), 'board_type'] = '科创板'
    trade_df.loc[trade_df.code.str.contains(".430|.830"), 'board_type'] = '新三板'
    trade_df.loc[trade_df.code.str.contains("bj."), 'board_type'] = '北交所'
    
    # ST的中文名称会随时间段变化(也存在没有对应中文名的情况)，通过日线的isST字段来判断训练，
    # trade_df.loc[trade_df.code_name.str.contains('ST'), 'board_type'] = 'ST'
    
    stock_board_dict = dict(zip(stock_df['code'], stock_df['board_type']))
    asset_dict = dict(zip(asset_df['board_type'], asset_df['price_limit_rate']))
    trade_df.loc[trade_df.board_type == '', 'board_type'] = trade_df.loc[trade_df.board_type == '', 'code'].map(stock_board_dict)
    trade_df['price_limit_rate'] = trade_df['board_type'].map(asset_dict)
    trade_df = trade_df.rename(columns={"tradestatus": "trade_status"})
    
    trade_df = utils_api_baostock.split_baostock_code(trade_df)
    
    # 结算周期
    trade_df['settlement_cycle'] = 1  # A股默认T+1
    
    trade_df = trade_df[['full_code', 'asset_code', 'market_code', 'code_name', 'board_type', 'price_limit_rate',
                         'trade_status', 'insert_timestamp']]
    return trade_df


def pipeline():
    trade_df = dwd_stock_base()
    utils_data.output_database(trade_df, filename='dwd_info_stock_incr_2', if_exists='replace')
    # trade_df.to_csv(f'{PATH}/_file/dwd_info_stock_incr.csv', index=False)


if __name__ == '__main__':
    pipeline()
