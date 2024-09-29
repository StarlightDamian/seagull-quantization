# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:23:41 2024

@author: awei
字符处理(character)
"""
import hashlib

def md5_file(data):
    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5

def md5_str(content):
    str_ma5 = hashlib.md5(content.encode(encoding='utf-8')).hexdigest()
    return str_ma5
