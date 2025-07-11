# -*- coding: utf-8 -*-
"""
@Date: 2025/6/24 23:41
@Author: Damian
@Email: zengyuwei1995@163.com
@File: bond_full.py
@Description:

"""
import pandas as pd
from seagull.utils import utils_data, utils_database, utils_log, utils_character
from seagull.utils.api import utils_api_baostock


def dwd_info_bond_adata():
    with utils_database.engine_conn("POSTGRES") as conn:
        adata_df = pd.read_sql('ods_info_bond_full_adata', con=conn.engine)

        adata_df = adata_df.rename(columns={'bond_code': 'bond_asset_code',
                                            'bond_name': 'bond_code_name',
                                            'stock_code': 'stock_asset_code',
                                            'short_name': 'stock_code_name',
                                            # 'sub_date'
                                            # 'issue_amount'
                                            # 'listing_date'
                                            # 'expire_date'
                                            'convert_price': 'prev_close',
                                            'market_id': 'bond_market_code',
                                            'stock_market_id': 'stock_market_code'})

    adata_df['bond_market_code'] = adata_df['bond_market_code'].map({'35': 'sz',
                                                                     '19': 'sh'})
    adata_df['stock_market_code'] = adata_df['stock_market_code'].map({'33': 'sz',
                                                                       '17': 'sh'})
    adata_df['stock_full_code'] = adata_df.stock_asset_code + '.' + adata_df.stock_market_code
    adata_df['bond_full_code'] = adata_df.bond_asset_code + '.' + adata_df.bond_market_code

    adata_df = adata_df[['bond_full_code', 'bond_asset_code', 'bond_market_code', 'bond_code_name',
                         'stock_full_code', 'stock_asset_code', 'stock_market_code', 'stock_code_name',
                         'prev_close', 'sub_date', 'issue_amount', 'listing_date', 'expire_date']]

    return adata_df


def dwd_info_bond_baostock():
    with utils_database.engine_conn("POSTGRES") as conn:
        baostock_df = pd.read_sql("select * from ods_info_stock_incr_baostock where type='4'", con=conn.engine)

    baostock_df = utils_api_baostock.split_baostock_code(baostock_df)

    baostock_df = baostock_df.rename(columns={'full_code': 'bond_full_code',
                                              'asset_code': 'bond_asset_code',
                                              'market_code': 'bond_market_code',
                                              'code_name': 'bond_code_name',
                                              'ipoDate': 'listing_date',
                                              'outDate': 'delisting_date',
                                              'status': 'trade_status'})
    return baostock_df


def dwd_info_stock_full():
    adata_df = dwd_info_bond_adata()
    baostock_df = dwd_info_bond_baostock()

    # 2. 打标 source
    baostock_df['source'] = 'baostock'
    adata_df['source'] = 'adata'

    # 在 merge 之前加上
    baostock_df = baostock_df.drop_duplicates('bond_full_code', keep='first')
    adata_df = adata_df.drop_duplicates('bond_full_code', keep='first')

    # 3. 合并：外连接，suffix 区分同名字段
    merged = pd.merge(
        adata_df,
        baostock_df,
        on=['bond_full_code'],
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
    merged['bond_asset_code'] = coalesce('bond_asset_code_bs', 'bond_asset_code_ad')
    merged['bond_market_code'] = coalesce('bond_market_code_bs', 'bond_market_code_ad')
    merged['bond_code_name'] = coalesce('bond_code_name_bs', 'bond_code_name_ad')
    merged['listing_date'] = coalesce('listing_date_bs', 'listing_date_ad')
    merged['settlement_cycle'] = 1  # A股默认T+1
    merged['price_limit_rate'] = 0.1
    merged = merged[['bond_full_code', 'bond_asset_code', 'bond_market_code', 'bond_code_name',
                     'listing_date', 'delisting_date', 'trade_status', 'price_limit_rate'
                     'stock_full_code', 'stock_asset_code', 'stock_market_code', 'stock_code_name',
                     'prev_close', 'sub_date', 'issue_amount', 'expire_date', 'settlement_cycle', 'source']]
    return merged


def pipeline():
    bond_df = dwd_info_stock_full()
    utils_data.output_database_large(bond_df,
                                     filename='dwd_info_bond_full', 
                                     if_exists='replace')


if __name__ == '__main__':
    pipeline()
