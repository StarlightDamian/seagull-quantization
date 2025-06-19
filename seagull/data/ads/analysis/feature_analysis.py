# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:42:44 2025

@author: awei
特征分析(feature_analysis)
"""

import os

import pandas as pd
import numpy as np

from seagull.settings import PATH
from analysis import icir3
from seagull.utils import utils_log, utils_thread

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

def calculate_close_rate(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.sort_values(by='date', ascending=True)
    df[['close_rate']] = df[['close_rate']].shift(-1)
    return df

def winsorize(df, n=3):
    """
    极值处理
    """
    mean = df.mean()
    std = df.std()
    upper = mean + n * std
    lower = mean - n * std
    return np.clip(df, lower, upper, axis=1)
    
def standardize(data):
    """
    标准化处理
    """
    return (data - data.mean()) / data.std()
    
if __name__ == "__main__":
    raw_df = pd.read_feather(f'{PATH}/_file/das_wide_incr_train.feather')
    
    
    df = raw_df.loc[raw_df.date=='2023-01-09',['alpha041','alpha042']]
    #df.alpha041.mean()==-671.059
    
    
    