# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:06:52 2024

@author: awei
数据结构(utils_structure)
"""

class StackStructure():
    '''
    功能：栈数据结构
    '''
    def __init__(self):
        self.stack = []
        self.stack_dict = {}
        
    def find_left_right_index(self, sentence, char_left, char_right):
        for idx,char in enumerate(sentence):
            if char == char_left:
                self.stack.append(idx)
            elif char == char_right:
                index_left = self.stack.pop()
                self.stack_dict[index_left] = idx#栈字典的键值对为{左：右}
        #print(self.stack_dict)
        index_pair_sorted = sorted([x for x in self.stack_dict.items()])#有序索引对
        index_left_list, index_right_list = [x for x in zip(*index_pair_sorted)]
        return index_left_list, index_right_list
    
    
# @lru_cache(maxsize=32)