# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:23:41 2024

@author: awei
字符处理(utils_character)
"""
import hashlib
import random
import string

def md5_file(data):
    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5

def md5_str(content):
    str_ma5 = hashlib.md5(content.encode(encoding='utf-8')).hexdigest()
    return str_ma5

def get_all_index_from_str(sentence, word):
    """
    功能：列表中子串的所有索引
    """
    try:
        return [n for n in range(len(sentence)) if sentence.find(word, n) == n]
    except Exception as e:
        print(str(e), sentence, word, 'utils_get_all_index_from_str异常')
        
def get_all_index_from_list(lst=None, item=''):
    """
    功能：列表中子串的所有索引
    """
    return [index for (index, value) in enumerate(lst) if value == item]

def generate_random_md5():
    #生成随机md5
    random_string = generate_random_string(16)  # 16字节随机字符串
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    return md5_hash

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

def is_contain_chinese(check_str):
    """
    功能:判断字符串中是否有中文
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

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

def find_specified_str(str_1, text):
    str_list = str_1.split('|')
    str_list = [x for x in str_list if x in text]
    return str_list

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

def round_floats(value):
    if isinstance(value, float):
        return round(value, 2)
    return value

# isinstance(5,int)