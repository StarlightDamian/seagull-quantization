# -*- coding: utf-8 -*-
"""
@Date: 2025/7/1 11:35
@Author: Damian
@Email: zengyuwei1995@163.com
@File: fund_full_efinance.py
@Description:
- ``'zq'``  : 债券类型基金
- ``'gp'``  : 股票类型基金
- ``'etf'`` : ETF 基金
- ``'hh'``  : 混合型基金
- ``'zs'``  : 指数型基金
- ``'fof'`` : FOF 基金
- ``'qdii'``: QDII 型基金
- ``None``  : 全部
"""
import efinance as ef
import pandas as pd

from seagull.utils import utils_data


def get_fund_full_data():
    fund_code_type_dict = {
        "zq": "债券类型基金",
        "gp": "股票类型基金",
        "etf": "ETF 基金",
        "hh": "混合型基金",
        "zs": "指数型基金",
        "fof": "FOF 基金",
        "qdii": "QDII 型基金"
    }

    # 一行搞定：拉取 → 打标签 → 合并 → 增加中文描述
    fund_df = (
        pd.concat(
            (
                ef.fund.get_fund_codes(ft=ft).assign(基金类型=ft)
                for ft in fund_code_type_dict
            ),
            ignore_index=True
        )
        .assign(基金类型描述=lambda df: df["基金类型"].map(fund_code_type_dict))
    )

    # 输出到数据库
    utils_data.output_database(
        fund_df,
        filename="ods_info_fund_full_efinance",
        if_exists="replace"
    )


if __name__ == '__main__':
    get_fund_full_data()
