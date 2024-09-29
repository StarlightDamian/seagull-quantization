# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:17:01 2022

@author: admin
工具包，常用工具模块，文件管理、时间管理(base_utils)
"""md5_str
import random
from datetime import datetime, timedelta
from functools import wraps
import string

import pandas as pd

#from __init__ import path


def run_time_decorator(func):
    """
    装饰器：计时器
    """
    @wraps(func)
    def wrapper(*args, **kw):
        time_stamp_start = datetime.timestamp(datetime.now())
        result = func(*args, **kw)
        time_stamp_end = datetime.timestamp(datetime.now())
        time_difference = '%.4f' % (time_stamp_end - time_stamp_start)
        print(f"文件路径： {__file__}\n函数名称： {func.__name__}\n运行耗时： {time_difference}秒\n------------------")
        return result
    return wrapper

# def run_time_decorator(func):
#     @wraps(func)
#     def wrapper(para, *args, **kw):
#         time_stamp_start = datetime.timestamp(datetime.now())
#         func(para, *args, **kw)
#         time_stamp_end = datetime.timestamp(datetime.now())
#         time_difference = '%.4f'%(time_stamp_end - time_stamp_start)
#         print(f"运行耗时：{time_difference}秒")
#         return func(para, time_difference)
#     return wrapper


def get_all_index_from_list(lst=None, item=''):
    """
    功能：列表中子串的所有索引
    """
    return [index for (index, value) in enumerate(lst) if value == item]


def get_all_index_from_str(sentence, word):
    """
    功能：列表中子串的所有索引
    """
    try:
        return [n for n in range(len(sentence)) if sentence.find(word, n) == n]
    except e:
        print(str(e), sentence, word, 'utils_get_all_index_from_str异常')


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


def find_file(path):
    """
    功能：返回地址下的所有文件
    输出：文件名称列表
    """
    for root, dirs, files in os.walk(path):
        return files


def lm_get_cmd_result(cmd):
    """
    功能：获取shell语句的返回值
    输入：shell语句
    """
    content = os.popen(cmd).read()
    return content


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


def build_random_str(len_str=6):
    """
    功能：生成随机字符串，kafka的KafkaConsumer.group_id属性需要每次不一样
    输入：len_str：字符串的长度
    """
    char_pool = 'qwertyuiopasdfghjklzxcvbnm123456789'  # 字符池
    char_list = [random.choice(char_pool) for idx in list(range(len_str))]
    return ''.join(char_list)


def build_index_pair_2(input_list):
    """
    功能：生成索引对
    
    输入示例：[6, 12, 19]
    输出示例：[[0,6],[6,12],[12,19],[19,10000]]
    
    输入示例2：[]
    输出示例2：[[0,10000]]
    """
    input_list = [0]+input_list+[10000]
    output_list = []
    for idx in list(range(len(input_list)-1)):
        output_list.append([input_list[idx], input_list[idx+1]])
    return output_list


def accumulator(input_list):
    """
    功能:累加器
    输入示例：[2, 5, 6]
    输出示例：[0, 2, 7]
    """
    output_list = [0]
    add = 0
    for idx, num in enumerate(input_list[:-1]):
        add += num
        output_list.append(add)
    return output_list

def accumulator_int_and_list(input_list, base_number):
    """
    功能:累加器
    输入示例：[2, 5, 6], 5
    输出示例：[5, 7, 10]
    """
    output_list = []
    for num in input_list:
        output_list.append(base_number)
        base_number += num
    return output_list

def accumulator_index_list(index_list, len_list):
    """
    功能:长索引累加器
    输入示例：index_list=[0,1,2,0,1,2],
            len_list=[2, 5, 6,1,5,6]
    输出示例：[0, 2, 7,0,1,6]
    """
    output_list, add_list = [], []
    for idx, index in enumerate(index_list):
        if (index == 0) and (idx != 0):
            output_list += accumulator(add_list)
            add_list = []
        add_list.append(len_list[idx])
    output_list += accumulator(add_list)
    return output_list


def fake_dtype(ws_pos, columns):
    """
    功能:伪类型转换
    """
    ws_pos = ws_pos.fillna('')
    for column in columns:
        ws_pos[column] = [x if '.' not in x else x[:-2] for x in ws_pos[column].tolist()]
    return ws_pos


def is_contain_chinese(check_str):
    """
    功能:判断字符串中是否有中文
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


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


def parenthesis_match(entity_one_polic):
    """
    功能:给左括号配对右括号，考虑各种溢出问题
    输出示例：[(22, 42),
             (139, 162),
             (449, 568),
             (900, 10000)]
    """
    max_index = 10000
    left_index_list = entity_one_polic[entity_one_polic.word == '('].index_start.tolist()
    right_index_list = entity_one_polic[entity_one_polic.word == ')'].index_start.tolist() + [max_index]
    len_right_index_list = len(right_index_list)
    output_pair_list = []
    
    for left_index in left_index_list:
        right_index = len_right_index_list - sum([int(x) > int(left_index) for x in right_index_list])
        output_pair_list.append((left_index, right_index_list[right_index]))
    return output_pair_list


import hashlib
# reload(sys)
# sys.setdefaultencoding('utf-8')


def md5_file(data):
    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5


def md5_str(content):
    str_ma5 = hashlib.md5(content.encode(encoding='utf-8')).hexdigest()
    return str_ma5

def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def generate_random_md5():
    #生成随机md5
    random_string = generate_random_string(16)  # 16字节随机字符串
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    return md5_hash
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


class NumberStrToInt:
    def __init__(self):
        self.number_dict = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000, '点': '.'}
        self.number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
        self.number_int_dict = {x[1]:x[0] for x in self.number_dict.items()}
    
    def transformation_int_to_str(self, int_add):
        number_str = ''
        # print(f'int_add:{int_add}')
        int_add = float(int_add)
        
        flag_int = int_add//100000000
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + '亿'
            int_add -= flag_int*100000000
            
        flag_int = int_add//10000000
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + '千万'
            int_add -= flag_int*10000000
        
        flag_int = int_add//1000000
        if flag_int >= 1:
            if '万'in number_str:
                number_str = number_str.replace('万', '')
            number_str += self.number_int_dict.get(flag_int) + '百万'
            int_add -= flag_int*1000000
        
        flag_int = int_add//100000
        if flag_int >= 1:
            if '万' in number_str:
                number_str = number_str.replace('万', '')
            number_str += self.number_int_dict.get(flag_int) + '十万'
            int_add -= flag_int*100000
            
        flag_int = int_add//10000
        if flag_int >= 1:
            if '万' in number_str:
                number_str = number_str.replace('万', '')
            number_str += self.number_int_dict.get(flag_int) + '万'
            int_add -= flag_int*10000
        
        flag_int = int_add//1000
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + '千'
            int_add -= flag_int*1000
        
        flag_int = int_add//100
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + '百'
            int_add -= flag_int*100
        
        flag_int = int_add//10
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + '十'
            int_add -= flag_int*10
            
        flag_int = int_add//1
        if flag_int >= 1:
            number_str += self.number_int_dict.get(flag_int) + ''
            int_add -= flag_int*1
        return number_str
        
    def number_int_to_str(self, text):
        output_str_money = ''
        int_add = ''
        for word in text+'元':
            if word in self.number_list:
                int_add += word
            elif (word not in self.number_list) and (int_add != ''):
                number_str = self.transformation_int_to_str(int_add)
                output_str_money += number_str+word
                int_add = ''
            elif (word not in self.number_list) and (int_add == ''):
                output_str_money += word
        # print(output_str_money)
        return output_str_money
    
    def superposition_number_detection(self, money_int):
        """
        功能：叠加树检测
        示例数据：texts=['四五个人','两三个人']
        """
        if str(money_int) in self.number_list:
            return True
        else:
            return False
        
    def get_one_number(self, text):
        try:
            # print('text',text)
            if text == '':
                return ''
            out_money = ''
            money_int_prev = money_type_prev = None
            text = self.number_int_to_str(text)
            superposition_number_prev_flag = False  # 用于标记叠加数
            for idx, word in enumerate(text):
                money_int = self.number_dict.get(word)
                # print(money_int)
                if money_int:
                    superposition_number_now_flag = self.superposition_number_detection(money_int)
                    if not(superposition_number_prev_flag & superposition_number_now_flag):
                        money_type = money_int // 10
                        money_type_bool = money_type >= 1
                        if money_int_prev is None:
                            out_money += str(money_int)
                        elif (money_int_prev != money_int) and (money_type_bool is False):
                            out_money += f'+{money_int}'
                        elif money_type_bool is True:
                            if (money_type_prev is None) or (money_type < money_type_prev):
                                out_money += f'*{money_int}'
                            elif money_type > money_type_prev:
                                out_money = f'({out_money})*{money_int}'
                            money_type_prev = money_type
                        money_int_prev = money_int
                        superposition_number_prev_flag = superposition_number_now_flag
                    else:
                        superposition_number_prev_flag = False
                # print(out_money)
            return eval(out_money)
        except:
            return False
        return eval(out_money)
    
    def get_n_number(self, texts):
        return [self.get_one_number(text) for text in texts]

    @classmethod
    def classmethod_n_number(cls, texts):
        return cls().get_n_number(texts)


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
 # def gettime(self, date_str, result_len):
 #        """
 #        获取时间gettime
 #        :param date_str:
 #        :param result_len:
 #        :return:time
 #        """
 #        try:
 #            date = date_str.replace("\\", "")#替换转移字符
 #            date = self.regNum.sub(" ", date_str)
 #            date_list = date.split(" ")
 #            data_list_len = len(date_list)
 #            if (data_list_len == 1):
 #                date = date_list[0]
 #            elif (len(date_list[0]) == 4):
 #                date = date_list[0]
 #                for i in range(1,data_list_len):
 #                    if (len(date_list[i]) == 1):
 #                        date += '0' + date_list[i]
 #                    elif (len(date_list[i]) == 2):
 #                        date += date_list[i]
 #            date = date[0:14]
 #            if (not self.checktime(date)):
 #                return None
 #            date_len = len(date)
 #            if(date_len < result_len):
 #                date += (int(result_len) - date_len) * "0"
 #            return date[0:result_len]
 #        except Exception:
 #            pass

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
    texts = ['160万元', '三十五万', '五万三十', '三百万', '一万三千', '三千两百万五千元', '三千零五万', '五万三千', '三百三十五', '三十五', '三千五十']
    texts = ['赌资上千', '两三个人', '一万三千']
    number_str_to_int = NumberStrToInt()
    number_int_list = number_str_to_int.get_n_number(texts)
    print(number_int_list)

    print(NumberStrToInt().classmethod_n_number(texts))
