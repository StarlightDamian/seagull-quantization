# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:46 2024

@author: awei
"""

import functools

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


if __name__ == '__main__':
    result = example_function(3, 4)
    print(f"Result: {result}")