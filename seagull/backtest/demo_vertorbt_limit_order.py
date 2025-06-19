# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:09:10 2025

@author: awei
demo_vectorbt_limit_order
"""
import os

import pandas as pd

from __init__ import path
from utils import utils_database, utils_log, utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')


def filter_valid_data(raw_df, threshold_ratio=0.7):
    """
    过滤有效数据，保留有效数据占比大于或等于 threshold_ratio 的日期。

    :param raw_df: 输入的DataFrame，包含 'date' 列和多个股票价格列。
    :param threshold_ratio: 允许的最小有效数据比例，默认是 0.7 (即70%有效数据)。
    :return: 处理后的 DataFrame 只包含有效数据的日期和股票代码。
    """
    # 计算每行有效数据的数量 (非 NaN)，并筛选出有效数据占比 >= threshold_ratio 的行
    threshold = threshold_ratio * (raw_df.shape[1] - 1)  # 去掉 'date' 列
    valid_rows = raw_df.drop(columns='date').notna().sum(axis=1) >= threshold
    
    # 筛选符合条件的日期范围
    filtered_df = raw_df[valid_rows]
    
    # 获取有效数据的开始日期和结束日期
    start_date = filtered_df['date'].min()
    end_date = filtered_df['date'].max()

    print(f"开始日期：{start_date}")
    print(f"结束日期：{end_date}")

    # 通过有效日期筛选数据
    valid_df = raw_df[raw_df['date'].isin(filtered_df['date'])]

    # 筛选出没有缺失值的列（对于股票数据来说，不需要计算 missing_df 是否有缺失）
    missing_value_columns = valid_df.loc[:, valid_df.notna().all()].columns

    # 返回有效数据的日期和没有缺失值的列
    return valid_df[missing_value_columns].reset_index(drop=True)


if __name__ == '__main__':
    with utils_database.engine_conn('postgre') as conn:
        raw_df = pd.read_sql("dwd_ohlc_full_portfolio_daily_backtest", con=conn.engine)
    
    # 调用函数，设置有效数据比例为 0.7
    result_df = filter_valid_data(raw_df, threshold_ratio=0.7)
    
    # 输出结果
    print(result_df.shape)

# =============================================================================
#     utils_data.output_database(result_df,
#                                filename='dwd_ohlc_full_portfolio_daily_backtest_eff',
#                                if_exists='replace')
# =============================================================================
    with utils_database.engine_conn('postgre') as conn:
        result_df.to_sql('dwd_ohlc_full_portfolio_daily_backtest_eff',
                  con=conn.engine,
                  index=False,
                  if_exists='replace',
                  )
# =============================================================================
# def filter_valid_data(raw_df, threshold_ratio=0.7):
#     """
#     过滤有效数据，保留有效数据占比大于或等于 threshold_ratio 的日期。
#     
#     :param raw_df: 输入的DataFrame，包含 'date' 列和多个股票价格列。
#     :param threshold_ratio: 允许的最小有效数据比例，默认是 0.7 (即70%有效数据)。
#     :return: 处理后的 DataFrame 只包含有效数据的日期和股票代码。
#     """
#     # 1. 计算每行中有效数据的数量 (非 NaN)
#     valid_data_count = raw_df.drop(columns='date').notna().sum(axis=1)
#     
#     # 2. 设置允许的最小有效数据数量：threshold_ratio的比例
#     threshold = threshold_ratio * (raw_df.shape[1] - 1)  # raw_df.shape[1] 是总列数，减去 'date' 列
#     
#     # 3. 筛选出符合条件的行（有效数据 >= threshold）
#     valid_rows = valid_data_count >= threshold
#     
#     # 4. 筛选出符合条件的日期范围
#     filtered_df = raw_df[valid_rows]
#     
#     # 5. 获取开始日期和结束日期
#     start_date = filtered_df['date'].min()
#     end_date = filtered_df['date'].max()
#     
#     print(f"开始日期：{start_date}")
#     print(f"结束日期：{end_date}")
#     
#     # 6. 获取有效日期范围内的所有数据
#     valid_df = raw_df[raw_df['date'].isin(filtered_df['date'])]
#     
#     # 7. 找到没有缺失值的列
#     missing_df = valid_df.isna().sum() == 0  # 检查没有缺失值的列
#     missing_value_columns = missing_df[missing_df].index  # 获取没有缺失值的列名
#     
#     # 8. 返回只包含没有缺失值的列的 DataFrame
#     result_df = valid_df.loc[:, missing_value_columns]
#     
#     return result_df
# =============================================================================