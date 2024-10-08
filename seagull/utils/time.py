# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:17:01 2022

@author: admin
时间(time)
"""
import random
import string
from datetime import datetime, timedelta


import pandas as pd

def __gen_dates(b_date, days):
    day = timedelta(days=1)
    for i in range(days.days):
        yield b_date + day*i


def __date_list(start, end):
    start_datetime = datetime.strptime(start, '%Y-%m-%d')
    end_datetime = datetime.strptime(end, '%Y-%m-%d')
    end_datetime = (end_datetime+timedelta(days=1))  # 结尾要加一天
    data = []
    for day in __gen_dates(start_datetime, (end_datetime - start_datetime)):
        data.append(day)
    return data

def date_binary_list(start_date, end_date):
    date_datetime = __date_list(start_date, end_date)
    date_list = [str(x)[:10] for x in date_datetime]
    date_binary_pair_list = []
    for idx, date in enumerate(date_list):
        try:
            date_binary_pair_list.append([date, date_list[idx+1]])
        except:
            break
    return date_binary_pair_list


def date_replace_binary_replace_list(date_start_replace, date_end_replace):
    date_start = datetime.strptime(date_start_replace, '%Y%m%d').strftime('%F')
    date_end = datetime.strptime(date_end_replace, '%Y%m%d').strftime('%F')
    return date_binary_replace_list(date_start, date_end)


def date_binary_replace_list(date_start, date_end):
    date_start = datetime.strptime(date_start, '%Y%m%d').strftime('%F') if '-' not in date_start else date_start
    date_end = datetime.strptime(date_end, '%Y%m%d').strftime('%F') if '-' not in date_end else date_end
    date_binary_pair_list = date_binary_list(date_start, date_end)
    date_binary_replace_list = [[x[0].replace('-', ''), x[1].replace('-', '')] for x in date_binary_pair_list]
    return date_binary_replace_list


def run_many_days(date_start, date_end, func):
    """
    功能：按日期跑每日数据，然后拼接.既可以跑一天的数据，也可以兼容多日数据合并的需求
    输入：date_start：'2022-06-13'
        func：输出单日数据的函数块
    输出：data_many_days_list:每日数据组成的列表
    """
    data_many_days_list = [func(date_start_replace, date_end_replace) for (date_start_replace, date_end_replace) in date_binary_replace_list(date_start, date_end)]
    return data_many_days_list




def date_suffix(date_type='today'):
    """
    功能：获取今日日期后缀
    备注：如今天是2022年2月23日。则输出 _20220223_20220224
    """
    today = datetime.now().strftime('%Y%m%d')
    if date_type == 'today':
        tomorrow = (datetime.now()+timedelta(days=1)).strftime('%Y%m%d')
        return '_' + today + '_' + tomorrow
    elif date_type == 'yesterday':
        yesterday = (datetime.now()+timedelta(days=-1)).strftime('%Y%m%d')
        return '_' + yesterday + '_' + today


def today_date_range(date_type='today'):
    """
    功能：获取今日日期列表
    备注：如今天是2022年2月23日。则输出 ['2022-02-23', '2022-02-24']
    """
    today = datetime.now().strftime('%F')
    if date_type == 'today':
        tomorrow = (datetime.now()+timedelta(days=1)).strftime('%F')
        return [today, tomorrow]
    elif date_type == 'yesterday':
        yesterday = (datetime.now()+timedelta(days=-1)).strftime('%F')
        return [yesterday, today]

















# =============================================================================
# import hashlib
# texts=['织里南海路与北盛唐路交叉口，两辆轿车相撞，无人员受伤','志称其的女儿（童亚娟 15215876411 ）情绪不稳定，需']
# md5 = hashlib.md5()
# md5.update(str(texts).encode('utf-8'))
# print(md5.hexdigest())
# =============================================================================
def find_specified_str(str_1, text):
    # str_1 = '没事了|找到了|没事|取消报警|撤案|无事'
    # text = '2019年1月4日21时31分许接110指令称:度假区网球中心旁边处的桥东村87号(超市处进去),其女友发了自杀割腕的照片给其,其现在找到了其女友的住处不敢进去。接警后由民警孟良带领两名协警处警,经出警至现场了解:报警人杨德宇(321324198807141015_15000365254)与其女友吵架,导致其女友恶作剧,用豆瓣酱涂抹手腕模仿了割腕现象拍照给其男友,不是自杀。经民警调解,事态已经平息。'
    str_list = str_1.split('|')
    str_list = [x for x in str_list if x in text]
    return str_list


def output_excel(forecast_original, file_path):
    with pd.ExcelWriter(file_path) as writer:
        forecast_original.to_excel(writer, sheet_name='预测表', index=False)

def dict_inversion(input_dict):
    """
    功能：字典反转
    输入：input_dict = {'a':1, 'b':2, 'c':1}
    输出：output_dict = {1:['a','c'],2:['b']}
    """
    from collections import defaultdict
    reversed_dict = defaultdict(list)
    for key, value in input_dict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def output_txt(d2):
    with open(f'{path}/data/fight_and_make_trouble.txt', 'w') as f:
        f.write(str(d2.jjd_bh.unique().tolist()))

# =============================================================================
# class StackStructure():
#     '''
#     功能：栈数据结构
#     '''
#     def __init__(self):
#         self.stack = []
#         self.stack_dict = {}
#         
#     def find_left_right_index(self, sentence, char_left, char_right):
#         for idx,char in enumerate(sentence):
#             if char == char_left:
#                 self.stack.append(idx)
#             elif char == char_right:
#                 index_left = self.stack.pop()
#                 self.stack_dict[index_left] = idx#栈字典的键值对为{左：右}
#         #print(self.stack_dict)
#         index_pair_sorted = sorted([x for x in self.stack_dict.items()])#有序索引对
#         index_left_list, index_right_list = [x for x in zip(*index_pair_sorted)]
#         return index_left_list, index_right_list
# =============================================================================


def stackfind_left_right_index(sentence, char_left, char_right):
    stack = []
    stack_dict = {}
    for idx, char in enumerate(sentence):
        if char == char_left:
            stack.append(idx)
        elif char == char_right:
            index_left = stack.pop()
            stack_dict[index_left] = idx  # 栈字典的键值对为{左：右}
    # print(self.stack_dict)
    index_pair_sorted = sorted([x for x in stack_dict.items()])  # 有序索引对
    index_left_list, index_right_list = [x for x in zip(*index_pair_sorted)]
    return index_left_list, index_right_list

def text_to_text_pd(texts):
    text_pd = pd.DataFrame([range(len(texts)), texts]).T.rename(columns={0: 'xxzjbh', 1: 'text'}).fillna('')
    text_pd.xxzjbh = text_pd.xxzjbh.astype(str)
    return text_pd

def file_is_open(file_path):
    """
    功能：判断文件是否已打开
    """
    try:
        print(open(file_path, 'w'))
        return False
    except Exception as e:
        if '[Errno 13] Permission denied' in str(e):
            print(f'{file_path},该文件已打开！')
            return True
        else:
            return False

def time_range_h(date_start='2024-03-08', date_end='2024-05-24'):
    hourly_range = pd.date_range(start=date_start, end=date_end, freq='H')
    time_pd = pd.DataFrame(hourly_range, columns=['time_start'])
    time_pd['time_end'] = time_pd['time_start'] + pd.Timedelta(minutes=59, seconds=59)
    time_pd['time_str'] = time_pd['time_start'].dt.strftime('%Y%m%d%H%M%S')+'_'+time_pd['time_end'].dt.strftime('%Y%m%d%H%M%S')
    return time_pd

def rename_filename():
    import os
    
    # 指定目录
    directory = r'C:\example'
    
    # 获取目录下所有文件
    files = os.listdir(directory)
    
    # 过滤出所有的 CSV 文件并重命名
    for filename in files:
        if filename.endswith('.csv'):
            old_path = os.path.join(directory, filename)
            new_filename = '1_' + filename
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
    
    print("文件重命名完成。")

def date_plus_days(date_start = "2000-01-01", days=1):
    start_date = datetime.strptime(date_start, "%Y-%m-%d")
    date_n_days_later = start_date + timedelta(days=days)
    formatted_date = date_n_days_later.strftime('%Y-%m-%d')
    return formatted_date  # '2000-04-11'

def round_floats(value):
    if isinstance(value, float):
        return round(value, 2)
    return value

# isinstance(5,int)
# =============================================================================
# #多进程
# import os
# from multiprocessing import Pool
# CPU_NUM = os.cpu_count()
# 
# def count_num(num):
#     return num
# 
# with Pool(CPU_NUM) as p:
#     return_p = p.map(count_num, list(range(5)))
# print(return_p)
# =============================================================================

# 换文件名os.rename(f'{path}/data/4_property/{file}', f'{path}/data/4_property_horizontal/{file[:9]}horizontal_{file[9:]}')
# 统计数量还原dataframe
# dict_soild_pd.bq_zwm.value_counts().rename_axis('bq_zwm').reset_index(name = '统计_标签总量')
# ws_pos_one_polic['len_ws'] = ws_pos_one_polic.ws.astype(str).str.len()
# ner_pd = ner_pd.astype(str)
# import pinyin
# pinyin.get_initial('中国',delimiter='').upper()
# a[1].astype(str).str.slice(start=-1)datafrmae最后一个字符
# from functools import lru_cache
# @lru_cache(maxsize=32)

# isinstance(3,int)
# {x:consumer_value.get(x,'') for x in label_key_list}
# {x:consumer_value.get(y,'') for (x,y) in zip(risk_key_list,risk_value_list)}
# labels_dic = bq_horizontal[field_list].T[0].to_dict()#对于有N条分类标签。只获取第一条
# 常用短语句
# score_pd.groupby(['ws']).sum()#按照分词累加
# df.A.unique()#唯一值
# df.A.value_counts()#频率
# person_pd.duplicated('word',keep='first')#判断第一个
# person_pd.drop_duplicates('word',keep='first') 保存第一个
# (datetime.now()+timedelta(days=1)).strftime('%Y%m%d')#明天
# .str.contains
# [x for tem in list_2 for x in tem]#二维转换为一维
# ztk_asj_jqrh_ds['产品@'].dtype=='bool'
# a1=a1.sort_values(by='index_start',ascending=False)排序
# today = datetime.now().strftime("%F")
# yesterday = (datetime.now()+timedelta(days=-1)).strftime('%F')
# .reset_index(drop=True) 序号
# ws_pos_one_polic['len_ws'] = ws_pos_one_polic.ws.astype(str).str.len()
# cpu核心数
# from multiprocessing import cpu_count
# print(cpu_count())
# output1 = pd.merge(output,ztk_asj_jq_ds_jqrh,on='jjd_bh')左右都需要唯一
# path_file = 'C:/Users/admin/Desktop/GeoHzWg.json'
# with open(path_file,'r',encoding='utf-8') as load_f:
#     load_dict = json.load(load_f)

#df['date_column'] = pd.to_datetime(df['date_column'])
#df['year_column'] = df['date_column'].dt.year

if __name__ == '__main__':
    # print(get_data_binary_list('2022-04-03', '2022-04-07'))
    # print(get_data_binary_replace_list('20220403', '20220407'))
    ...
