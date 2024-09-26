# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:38:05 2023

@author: awei
应用层_蒙特卡洛方法_模拟交易
application_monte_carlo

industry = array(['银行', nan, '交通运输', '汽车', '房地产', '公用事业', '钢铁', '化工', '非银金融', '机械设备',
       '传媒', '国防军工', '建筑装饰', '通信', '综合', '休闲服务', '医药生物', '商业贸易', '食品饮料',
       '家用电器', '电子', '轻工制造', '电气设备', '农林牧渔', '计算机', '纺织服装', '有色金属', '采掘',
       '建筑材料'], dtype=object)
"""
import argparse
import pandas as pd
#from base import base_data_loading


from __init__ import path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-02-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    stock_industry_df = pd.read_csv(f'{path}/data/stock_industry.csv', encoding='gb18030')
    plate_df = stock_industry_df[stock_industry_df.industry.isin(['计算机'])]
    
    # day_df
    