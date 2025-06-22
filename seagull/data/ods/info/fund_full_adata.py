# -*- coding: utf-8 -*-
"""
@Date: 2025/6/16 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_full_adata.py
@Description: (ods/info/fund_full_adata)
@Update cycle: day
"""
import adata

from seagull.settings import PATH
from seagull.utils import utils_data


def ods_info_fund_full_data():
    df = adata.fund.info.all_etf_exchange_traded_info()
    utils_data.output_database(df,
                               filename='ods_info_fund_full_adata')


if __name__ == '__main__':
    ods_info_fund_full_data()
