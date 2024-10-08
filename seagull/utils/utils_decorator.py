# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:46 2024

@author: awei
"""

import functools
from functools import wraps
from datetime import datetime

def print_vars(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Variables in {func.__name__}:")
        for name, value in locals().items():
            if name != 'func' and name != 'args' and name != 'kwargs':
                print(f"  {name} = {value}")
        return result
    return wrapper

@print_vars
def example_function(a, b):
    c = a + b
    return c

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

if __name__ == '__main__':
    result = example_function(3, 4)
    print(f"Result: {result}")