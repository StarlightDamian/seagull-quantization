# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 11:14
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_3d_array.py
@Description: 
"""
import numpy as np
import pandas as pd

from seagull.utils import utils_database, utils_character, utils_log, utils_data, utils_thread


def dwd_ohlc_fund_full_daily_backtest():
    with utils_database.engine_conn("POSTGRES") as conn:
        fund_df = pd.read_sql("dwd_freq_incr_fund_daily", con=conn.engine)
    fund_df = fund_df[['date', 'full_code', 'close']]
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    fund_daily_backtest_df = fund_df.pivot(index='date', columns='full_code', values='close')
    utils_data.output_database(fund_daily_backtest_df, filename='dwd_freq_full_fund_daily_backtest',
                               index=True)


if __name__ == '__main__':
    # path = '/'
    # raw_df = pd.read_feather(f'{path}/_file/das_wide_incr_train_mini.feather')
    with utils_database.engine_conn("POSTGRES") as conn:
        # date 都在交易日
        # trading_day_df = pd.read_sql("select date from dwd_base_full_trading_day where trade_status=1", con=conn.engine)
        # trading_day = trading_day_df['date'].tolist()
        raw_df = pd.read_sql('dwd_ohlc_fund_incr_daily', con=conn.engine)
        # raw_df = raw_df[(raw_df.date.isin(trading_day))]

    features = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'price_chg', 'turn']

    # 假设 raw_df 已经按 date 排序好了
    raw_df['date'] = pd.to_datetime(raw_df['date'])

    # 1) 把 (date, full_code) 设为行索引，只留下我们需要的 features：
    dfm = raw_df.set_index(['date', 'full_code'])[features]

    # 2) 对每个 feature 一次性 unstack（一次循环）并堆叠成 3D：
    arr3d = np.stack(
        [dfm[feat].unstack().values for feat in features],
        axis=2
    )

    # 3) 保存一下标签
    dates_array = dfm.index.levels[0].values  # (n_dates,)
    codes_array = dfm.index.levels[1].values  # (n_codes,)

    # arr3d[:,:,features.index('low')]
