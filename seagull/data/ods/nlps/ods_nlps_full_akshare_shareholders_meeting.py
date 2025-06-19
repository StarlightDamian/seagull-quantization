# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:08:47 2025

@author: Damian

股东大会(ods_info_full_akshare_shareholders_meeting)

['代码', '简称', '股东大会名称', '召开开始日', '股权登记日', '现场登记日', '网络投票时间-开始日',
       '网络投票时间-结束日', '决议公告日', '公告日', '序列号', '提案', 'insert_timestamp']
"""

import akshare as ak

from seagull.settings import PATH
from seagull.utils import utils_data #utils_database, utils_log ,

if __name__ == '__main__':
    shareholders_meeting_df = ak.stock_gddh_em()
    utils_data.output_database(shareholders_meeting_df,
                               filename='ods_info_full_akshare_shareholders_meeting',
                               )