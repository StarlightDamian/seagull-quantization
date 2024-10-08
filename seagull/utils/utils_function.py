# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:03:47 2024

@author: awei
功能类(utils_function)
"""

def get_pinyin_initial():
    """
    功能：把一列中文转化为拼音
    """
    import pinyin
    import pandas as pd
    data = pd.read_csv('D:/semantic_analysis_police_information_v1.3.0/data/transformation_pinyin.csv', encoding='gb18030')
    field_list = data.field.tolist()
    pingyin_list = [pinyin.get_initial(x, delimiter="") for x in field_list]
    pingyin_list = [x.replace('/', '_') if '/' in x else x for x in pingyin_list]
    data['pinyin'] = pingyin_list
    data.to_csv('D:/semantic_analysis_police_information_v1.3.0/data/output_transformation_pinyin.csv')