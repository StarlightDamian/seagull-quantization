# -*- coding: utf-8 -*-
"""
@Date: 2024/10/21 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: index_incr_efinance.py
@Description: 全球指数(ods/ohlc/stock_incr_efinance_global_index)
@Update cycle: day

全球指数的每日获取时间周期和A股不一样
macro_sz399101_diff
macro_sh000001_diff

sh.000002, 上证A股指数,432298594654
sz.399107, 深证A股指数,356732648847
sh.000001,上证综合指数,357057491972
sz.399106, 深证综合指数,432390342915
"""
from seagull.utils.api import utils_api_efinance
from seagull.utils import utils_data

index_dict = {'HSI': '恒生指数',  
              'HSTECH': '恒生科技指数',
              'TQQQ': '三倍做多纳斯达克100ETF',
              'HXC': '纳斯达克中国金龙指数',
              'SPX': '标准普尔500指数',
              'YANG': '三倍做空富时中国ETF-Direxion',
              }

if __name__ == '__main__':
    full_code_arr = index_dict.keys()
    df = utils_api_efinance.efinance_codelist2stock(full_code_arr)
    utils_data.output_database(df,
                               filename='ods_ohlc_index_incr_efinance',
                               if_exists='append')
