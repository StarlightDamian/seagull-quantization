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

listing_date, 上市日期
delisting_date, 退市日
"""
import os
import pandas as pd
from seagull.settings import PATH
from seagull.utils import utils_data, utils_database, utils_log, utils_character
from seagull.utils.api import utils_api_baostock


def dwd_info_stock_baostock():
    # ods_info_stock_incr_baostock.columns = ['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status',
    # 'insert_timestamp']
    with utils_database.engine_conn("POSTGRES") as conn:
        baostock_df = pd.read_sql("select * from ods_info_stock_incr_baostock where type='1'", con=conn.engine)

    baostock_df = utils_api_baostock.split_baostock_code(baostock_df)

    baostock_df = baostock_df.rename(columns={'status': 'trade_status',
                                              'ipoDate': 'listing_date',
                                              'outDate': 'delisting_date',
                                              })
    return baostock_df


def dwd_info_stock_adata():
    with utils_database.engine_conn("POSTGRES") as conn:
        adata_df = pd.read_sql("ods_info_stock_incr_adata", con=conn.engine)

    adata_df = adata_df.rename(columns={'exchange': 'market_code',
                                        'stock_code': 'asset_code',
                                        'short_name': 'code_name',
                                        'list_date': 'listing_date'})

    adata_df['market_code'] = adata_df['market_code'].str.lower()
    adata_df['full_code'] = adata_df['asset_code'] + '.' + adata_df['market_code']
    return adata_df


def dwd_info_stock_full():
    # 1. 读两张表
    baostock_df = dwd_info_stock_baostock()
    adata_df = dwd_info_stock_adata()

    # 2. 打标 source
    baostock_df['source'] = 'baostock'
    adata_df['source'] = 'adata'

    # 在 merge 之前加上
    baostock_df = baostock_df.drop_duplicates('full_code', keep='first')
    adata_df = adata_df.drop_duplicates('full_code', keep='first')

    # 3. 合并：外连接，suffix 区分同名字段
    merged = pd.merge(
        adata_df,
        baostock_df,
        on=['full_code'],
        how='outer',
        suffixes=('_ad', '_bs'),
        indicator=True
    )

    # 4. 生成 final_source
    merged['source'] = merged['_merge'].map({
        'both':      'adata,baostock',
        'left_only': 'adata',
        'right_only': 'baostock'
    })

    # 5. 对重复字段做“非空优先”合并
    def coalesce(bs_col, ad_col=None):
        # 如果 ad_col 参数是真正提供了某列名，就做 fillna；否则直接返回 bs_col
        if ad_col is None:
            return merged[bs_col]
        return merged[bs_col].fillna(merged[ad_col])

    print(merged.columns)
    merged['code_name'] = coalesce('code_name_bs', 'code_name_ad')
    merged['asset_code'] = coalesce('asset_code_bs', 'asset_code_ad')
    merged['market_code'] = coalesce('market_code_bs', 'market_code_ad')
    merged['listing_date'] = coalesce('listing_date_bs', 'listing_date_ad')
    # merged['delisting_date'] = coalesce('delisting_date_bs', 'delisting_date_ad')
    merged['trade_status'] = coalesce('trade_status', None)  # 若 adata 没此字段
    # merged['listing_date'] = coalesce('listing_date', None)  # 来自 adata_df

    with utils_database.engine_conn("POSTGRES") as conn:
        asset_df = pd.read_sql("ods_base_full_asset_details", con=conn.engine)

    merged[["board_type", "price_limit_rate", "date_max"]] = ''
    fc = merged['full_code']
    # 主板：上交所 60 开头或深交所 00 开头
    cond_main = (fc.str.startswith('60') & fc.str.endswith('.sh')) | \
                (fc.str.startswith('00') & fc.str.endswith('.sz'))
    merged.loc[cond_main, 'board_type'] = '主板'

    # 创业板：深交所 300、301 开头
    cond_cyb = fc.str.startswith(('300', '301')) & fc.str.endswith('.sz')
    merged.loc[cond_cyb, 'board_type'] = '创业板'

    # 科创板：上交所 688、689 开头
    cond_kcb = fc.str.startswith(('688', '689')) & fc.str.endswith('.sh')
    merged.loc[cond_kcb, 'board_type'] = '科创板'

    # 新三板（北交所/精选层）：430、831 开头，假设后缀也是 '.sz'
    cond_nsb = fc.str.startswith(('430', '831')) & fc.str.endswith('.sz')
    merged.loc[cond_nsb, 'board_type'] = '新三板'

    merged.loc[fc.str.endswith('.bj'), 'board_type'] = '北交所'

    # ST的中文名称会随时间段变化(也存在没有对应中文名的情况)，通过日线的isST字段来判断训练，
    # trade_df.loc[trade_df.code_name.str.contains('ST'), 'board_type'] = 'ST'

    # stock_board_dict = dict(zip(merged['code'], merged['board_type']))
    # merged.loc[merged.board_type == '', 'board_type'] = merged.loc[merged.board_type == '', 'code'].map(
    #     stock_board_dict)

    asset_dict = dict(zip(asset_df['board_type'], asset_df['price_limit_rate']))
    merged['price_limit_rate'] = merged['board_type'].map(asset_dict)

    # 结算周期, A股默认T+1
    merged['settlement_cycle'] = 1
    merged.loc[merged['market_code'] == 'bj', 'settlement_cycle'] = 0

    # 6. 选留最终需要的列
    final = merged[['full_code', 'asset_code', 'market_code', 'code_name', 'trade_status', 'listing_date',
                    'delisting_date', 'date_max', 'board_type', 'price_limit_rate', 'settlement_cycle', 'source']]

    return final


def pipeline():
    stock_df = dwd_info_stock_full()
    utils_data.output_database(stock_df, filename='dwd_info_stock_full', if_exists='replace')


if __name__ == '__main__':
    pipeline()



