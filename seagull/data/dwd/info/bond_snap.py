# -*- coding: utf-8 -*-
"""
@Date: 2025/6/24 23:41
@Author: Damian
@Email: zengyuwei1995@163.com
@File: bond_snap.py
@Description: 
"""


def dwd_info_nrtd_bond_base(self):
    with utils_database.engine_conn("POSTGRES") as conn:
        bond_base_df = pd.read_sql('ods_info_nrtd_adata_bond_base', con=conn.engine)

        bond_base_df = bond_base_df.rename(columns={'bond_code': 'bond_asset_code',
                                                    'bond_name': 'bond_code_name',
                                                    'stock_code': 'stock_asset_code',
                                                    'short_name': 'stock_code_name',
                                                    # 'sub_date'
                                                    # 'issue_amount'
                                                    # 'listing_date'
                                                    # 'expire_date'
                                                    'convert_price': 'prev_close',
                                                    'market_id': 'market_code',
                                                    'stock_market_id': 'stock_market_id',
                                                    })

    bond_base_df['market_code'] = bond_base_df['market_code'].map({'35': 'SZ',
                                                                   '19': 'SH'})
    bond_base_df['stock_market_code'] = bond_base_df['stock_market_code'].map({'35': 'SZ',
                                                                               '19': 'SH'})
    bond_base_df['stock_full_code'] = bond_base_df.stock_market_id + '.' + bond_base_df.stock_asset_code
    bond_base_df['bond_full_code'] = bond_base_df.bond_market_id + '.' + bond_base_df.bond_asset_code
    utils_data.output_database(bond_base_df, 'dwd_info_nrtd_bond_base')