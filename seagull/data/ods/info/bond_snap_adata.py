# -*- coding: utf-8 -*-
"""
@Date: 2024/8/27 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: bond_snap_adata.py
@Description: 获取所有A股市场的可转换债券代码信息	(ods/info/bond_snap_adata)
@Update cycle: day
https://github.com/1nchaos/adata/blob/ba93ab86af7256407db069668e84b54b5fd6dbd7/adata/bond/info/bond_code.py#L36
# bond.info.all_convert_code()
bond-conversion
不能开国外VPN获取数据
"""

import copy
import requests

import pandas as pd
from adata.bond.info.bond_code import BondCode
from adata.common.headers import ths_headers

from seagull.settings import PATH
from seagull.utils import utils_data


class OdsInfoBondSnapAdata(BondCode):
    """
    债券代码
    """
    def __init__(self) -> None:
        super().__init__()
        
    def ods_info_bond_snap_adata_ths(self):
        """
        获取同花顺可转换债券列表
        web： http://data.10jqka.com.cn/ipo/kzz/
        :return 可转债列表
        ['bond_code','bond_name','stock_code','short_name','sub_date','issue_amount','listing_date','expire_date',
        'convert_price', 'market_id', 'stock_market_id']
        """
        COLUMNS = ['bond_code', 'bond_name', 'stock_code', 'short_name', 'sub_date', 'issue_amount', 'listing_date',
                   'expire_date', 'convert_price', 'market_id', 'stock_market_id'] 
        # 1. 请求市场排名的 url
        api_url = "https://data.10jqka.com.cn/ipo/kzz/"
        # 2. 设置请求头
        headers = copy.deepcopy(ths_headers.text_headers)
        headers['Host'] = 'data.10jqka.com.cn'
        headers['Referer'] = 'http://data.10jqka.com.cn/ipo/bond/'
        res = requests.get(api_url, headers=headers, proxies={})
        res_json = res.json()
        if res.status_code != 200 or res_json['status_msg'] != 'ok':
            return pd.DataFrame(columns=COLUMNS)
        
        # 3. 解析数据
        data = res_json['list']
        
        # 4. 封装数据
        df = pd.DataFrame(data=data).rename(columns={'price': 'convert_price',
                                                     'issue_total': 'issue_amount',
                                                     'name': 'short_name',
                                                     'code': 'stock_code'})[COLUMNS]
        # 5. 数据清洗
        df['issue_amount'] = df['issue_amount'].astype(float) * 100000000
        return df

    def pipeline(self):
        convertibles_base_df = self.ods_info_bond_snap_adata_ths()
        utils_data.output_database(convertibles_base_df,
                                   'ods_info_bond_snap_adata',
                                   if_exists='replace')


if __name__ == '__main__':
    ods_info_bond_snap_adata = OdsInfoBondSnapAdata()
    ods_info_bond_snap_adata.pipeline()

# =============================================================================
#  {'sub_date': '2017-03-17',
#   'bond_code': '113011',
#   'bond_name': '光大转债',
#   'code': '601818',
#   'name': '光大银行',
#   'sub_code': '783818',  # 暂不需要
#   'share_code': '764818',  # 暂不需要
#   'sign_date': '2017-03-23',  # 暂不需要
#   'plan_total': '300.000000',  # 暂不需要
#   'issue_total': '300.000000',
#   'issue_price': '100.0000',  # 暂不需要
#   'success_rate': '0.49992510',  # 暂不需要
#   'listing_date': '2017-04-05',
#   'expire_date': '2023-03-17',
#   'price': '3.350',
#   'quota': '0.75300',  # 暂不需要
#   'number': '0',  # 暂不需要
#   'market_id': '19', 
#   'stock_market_id': '17' } 
# =============================================================================
