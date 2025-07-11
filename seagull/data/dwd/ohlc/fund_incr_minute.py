# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 16:58
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_incr_minute.py
@Description: 
"""
import pandas as pd

from seagull.utils import utils_database, utils_character, utils_log, utils_data, utils_thread


def get_dwd_ohlc_fund_incr_minute():
    # 清洗efinance的get_quote_history()接口数据
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql('ods_ohlc_fund_incr_efinance_minute', con=conn.engine)
        fund_info_df = pd.read_sql('dwd_info_fund_full', con=conn.engine)
    fund_info_df = fund_info_df[['market_code', 'full_code', 'asset_code']]

    fund_df = fund_df.rename(columns={'股票名称': 'code_name',
                                      '股票代码': 'asset_code',
                                      '日期': 'date',
                                      '开盘': 'open',
                                      '收盘': 'close',
                                      '最高': 'high',
                                      '最低': 'low',
                                      '成交量': 'volume',
                                      '成交额': 'amount',
                                      '振幅': 'amplitude',  # new
                                      '涨跌幅': 'pct_chg',
                                      '涨跌额': 'price_chg',  # new
                                      '换手率': 'turn'
                                       })

    fund_df['freq_code'] = 5
    fund_df['adj_code'] = 0  # adj_code = {0: None, 1: "pre", 2: "post"}
    # primary_key主键不参与训练，用于关联对应数据. code_name因为是最新的中文名,ST不具有长期意义
    fund_df['time'] = pd.to_datetime(fund_df['date']).dt.strftime("%Y%m%d%H%M%S")
    fund_df['date'] = fund_df['date'].str[:10]
    fund_df = pd.merge(fund_df, fund_info_df, on='asset_code')

    fund_df['primary_key'] = (fund_df['time'].astype(str) +
                              fund_df['full_code'].astype(str) +
                              fund_df['freq_code'].astype(str) +
                              fund_df['adj_code'].astype(str)
                              ).apply(utils_character.md5_str)  # md5（时间、代码、频率、复权）
    fund_df = fund_df[['full_code', 'asset_code', 'market_code', 'code_name', 'date', 'time', 'open', 'high', 'low',
                       'close', 'volume', 'amount', 'amplitude', 'pct_chg', 'price_chg', 'turn', 'freq_code',
                       'adj_code', 'primary_key']]

    utils_data.output_database_large(fund_df,
                                     filename='dwd_ohlc_fund_incr_minute',
                                     if_exists='replace')


if __name__ == '__main__':
    get_dwd_ohlc_fund_incr_minute()
