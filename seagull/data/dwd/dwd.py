# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei

"""
import argparse
from loguru import logger

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_data, utils_character

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='证券代码', help='["证券代码", "证券标签"]')
    args = parser.parse_args()
    
    dwd_data = dwdData()
    dwd_data.pipline(args.data_type)
