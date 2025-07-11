# -*- coding: utf-8 -*-
"""
@Date: 2025/7/11 14:46
@Author: Damian
@Email: zengyuwei1995@163.com
@File: stock_incr_minute.py
@Description: 
"""


def stock_minute(self):
    with utils_database.engine_conn("POSTGRES") as conn:
        baostock_stock_sh_sz_minute_df = pd.read_sql("ods_ohlc_incr_baostock_stock_sh_sz_minute", con=conn.engine)
    clean_stock_sh_sz_df = self.clean_baostock_query_history_k_data_plus(baostock_stock_sh_sz_minute_df)

    clean_asset_minute_df = pd.concat([clean_stock_sh_sz_df], axis=0)
    utils_data.output_database(clean_asset_minute_df,
                               filename='dwd_ohlc_incr_stock_minute',
                               dtype={'primary_key': String,
                                      'date': String,
                                      'time': String,
                                      'asset_code': String,
                                      'open': Float,
                                      'high': Float,
                                      'low': Float,
                                      'close': Float,
                                      'volume': Numeric,
                                      'amount': Numeric,
                                      })