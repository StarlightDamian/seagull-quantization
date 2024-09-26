# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:30:03 2024

@author: awei
数据工具包(base_data)
"""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from __init__ import path
from base import base_utils, base_connect_database, base_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = base_log.logger_config_local(f'{path}/log/{log_filename}.log')

def output_database(df, filename, chunksize=1000_000, if_exists='append', dtype=None, index=False):
    """
    :param conn:连接方式
    """
    if not df.empty:
        df['insert_timestamp'] = datetime.now().strftime("%F %T")
        logger.success('Writing to database started.')
        with base_connect_database.engine_conn('postgre') as conn:
            df.to_sql(filename,
                      con=conn.engine,
                      index=index,
                      if_exists=if_exists,
                      chunksize=chunksize,
                      dtype=dtype)
        logger.success('Writing to database conclusion-succeeded.')
    else:
        logger.info('Writing to database conclusion-failed.')

def output_local_file(df, filename, if_exists='skip', encoding='gbk', file_format='csv', filepath=None):
    filepath = filepath if filepath else f"{path}/data/{filename}.{file_format}"
    if if_exists=='overwrite':
            df.to_csv(filepath, encoding=encoding, index=False)
    elif os.path.exists(filename) and if_exists=='append':
            df.to_csv(filepath, encoding=encoding, index=False, mode='a', header=False)
    elif not os.path.exists(filename) and if_exists=='skip':
        df.to_csv(filepath, encoding=encoding, index=False)
    else:
        ...
            
def maximum_date(table_name, field_name='date', sql=None):
    try:
        with base_connect_database.engine_conn('postgre') as conn:
            if sql:
                max_date = pd.read_sql(sql, con=conn.engine)
            else:
                max_date = pd.read_sql(f"SELECT max({field_name}) FROM {table_name}", con=conn.engine)
        max_date = max_date.values[0][0]
        logger.info(f'max_date: {max_date}')
    except:
        logger.error('Exception in querying database maximum date')
        max_date = '1990-01-01'
    finally:
        next_day = datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=1)
        date_start = next_day.strftime('%Y-%m-%d')
        logger.info(f'date_start: {date_start}')
        return date_start

def feather_file_merge(date_start, date_end):
    date_binary_pair_list = base_utils.date_binary_list(date_start, date_end)
    feather_files = [f'{path}/data/day/{date_binary_pair[0]}.feather' for date_binary_pair in date_binary_pair_list]
    #print(feather_files)
    dfs = [pd.read_feather(file) for file in feather_files if os.path.exists(file)]
    feather_df = pd.concat(dfs, ignore_index=True)
    return feather_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-02-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    date_range_df = feather_file_merge(args.date_start, args.date_end)
    print(date_range_df)
        