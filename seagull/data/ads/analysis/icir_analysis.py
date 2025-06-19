# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:10:22 2025

@author: awei
对ICIR结果进行分析(icir_analysis)
"""

import os

import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_log, utils_database

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


if __name__ == "__main__":
    with utils_database.engine_conn("POSTGRES") as conn:
        raw_df = pd.read_sql("ads_info_incr_icir", con=conn.engine)
    print(raw_df)
    abs_df = raw_df
    abs_df[['mean_ic', 'ir']] = abs_df[['mean_ic', 'ir']].abs()


    d_df = abs_df[abs_df.freq=='1d']
    
    d_df[['mean_ic', 'ir']] = d_df[['mean_ic', 'ir']].abs()
    d_df.groupby('remark').mean_ic.mean()
    d_df.groupby('remark').ir.mean()
    
    output_df = d_df.pivot_table(index='feature_name', columns='remark', values='mean_ic')
    output_df['max'] = output_df.idxmax(axis=1)
    output_df = output_df.round(5)
    output_df.to_csv(f'{PATH}/_file/analysis_ic.csv', index=True)
    
    output_df = d_df.pivot_table(index='feature_name', columns='remark', values='ir')
    output_df['max'] = output_df.idxmax(axis=1)
    output_df = output_df.round(5)
    output_df.to_csv(f'{PATH}/_file/analysis_ir.csv', index=True)
    
    output_df = abs_df.pivot_table(index='feature_name', columns='freq', values='mean_ic')
    output_df['max'] = output_df.idxmax(axis=1)
    output_df.to_csv(f'{PATH}/_file/analysis_freq.csv', index=True)
    
    