# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:41:49 2025

@author: Damian

重大合同(ods_info_incr_akshare_contract)

['序号', '股票代码', '股票简称', '签署主体', '签署主体-与上市公司关系', '其他签署方', '其他签署方-与上市公司关系',
       '合同类型', '合同名称', '合同金额', '上年度营业收入', '占上年度营业收入比例', '最新财务报表的营业收入', '签署日期',
       '公告日期']
"""

import akshare as ak




import akshare as ak

from seagull.settings import PATH
from seagull.utils import utils_data #utils_database, utils_log ,

if __name__ == '__main__':
    contract_df = ak.stock_zdhtmx_em(start_date="20220819", end_date="20250819")
    utils_data.output_database(contract_df,
                               filename='ods_info_incr_akshare_contract',
                               )