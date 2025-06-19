# -*- coding: utf-8 -*-
"""
@Date: 2024/7/20 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_data.py
@Description: 数据/文件管理
"""
import argparse
import os
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd

from __init__ import path
from utils import utils_time, utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')
import csv


def map_dtype_to_postgres(pandas_dtype):
    """
    将 pandas 数据类型映射到 PostgreSQL 数据类型
    :param pandas_dtype: pandas 数据类型
    :return: 对应的 PostgreSQL 数据类型
    """
    dtype_map = {
        'int64': 'BIGINT',
        'float64': 'float8',#'DOUBLE PRECISION',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'datetime64[ns, UTC]': 'TIMESTAMP'
    }
    return dtype_map.get(str(pandas_dtype), 'TEXT')  # 默认返回 TEXT 类型


def output_database_large(df, filename, if_exists='append'):
    conn = utils_database.psycopg2_conn()
    cursor = conn.cursor()
    # Step 2: 如果 if_exists 为 'replace'，先删除表再重新创建
    if if_exists == 'replace':
        logger.info(f"Table {filename} exists, dropping and recreating it.")
        cursor.execute(f"DROP TABLE IF EXISTS {filename};")
        conn.commit()
     
    df[df.select_dtypes(include=['float']).columns] = df.select_dtypes(include=['float']).fillna(0)
    df['insert_timestamp'] = datetime.now().strftime("%F %T")
    
    # Step 1: 确保表存在，如果表不存在则创建
    cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = '{filename}'
        );
    """)
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        # 动态生成创建表的 SQL 语句
        #columns = ', '.join([f"{col} {dtype}" for col, dtype in zip(df.columns, ['TEXT']*len(df.columns))])
        columns = ', '.join([f"{col} {map_dtype_to_postgres(df[col].dtype.name)}" for col in df.columns])
        logger.info(columns)
        create_table_query = f"""
            CREATE TABLE {filename} (
                {columns}
            );
        """
        cursor.execute(create_table_query)
        conn.commit()
    
    # Step 2: 确保字段一致，如果有缺失的列，添加 NULL 值
    # 获取目标表的列名
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{filename}' ORDER BY ordinal_position;")
    existing_columns = [col[0] for col in cursor.fetchall()]
    
    # 将 DataFrame 的列与表的列进行对比，添加缺失列
    missing_columns = set(existing_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = np.nan  # 为缺失的列添加空值（NULL）
    
    # 重新排列 DataFrame 列的顺序，确保顺序与表中的列一致
    df = df[existing_columns]
    
    # Step 3: 将 DataFrame 转换为 CSV 格式并写入数据库
    csv_buffer = StringIO()
    #df = df.where(pd.notnull(df), None)  # Replace NaN with None to treat as NULL
    df.to_csv(csv_buffer,
              index=False,
              #quoting=csv.QUOTE_MINIMAL,
              #escapechar='\\',
              header=False
              )
    csv_buffer.seek(0)
    
    # 使用 COPY 命令将数据写入数据库
    cursor.copy_from(csv_buffer, filename, sep=',')
    conn.commit()
    
    # 关闭连接
    cursor.close()
    conn.close()


def output_database_mini(df, filename, chunksize=50_000, if_exists='append', dtype=None, index=False, method="multi"):
    try:
        logger.info('Writing to database started.')
        with utils_database.engine_conn('postgre') as conn:
            df.to_sql(filename,
                      con=conn.engine,
                      index=index,
                      if_exists=if_exists,
                      chunksize=chunksize,
                      dtype=dtype,
                      method=method
                      )
        logger.success('Writing to database conclusion-succeeded.')
    except Exception as e:
        logger.error(f'Writing to database conclusion-failed: {e}')


def output_database(df, **kwargs):
    """
    :param conn:连接方式
    """
    if not df.empty:
        df['insert_timestamp'] = datetime.now().strftime("%F %T")
        len_df = df.shape[0]
        logger.info(f'The DataFrame has {len_df} rows.')
        chunk_size = 50_000
        for start in range(0, len_df, chunk_size):
            df_chunk = df.iloc[start: start+chunk_size]
            if start!=0:
                kwargs['if_exists']='append'
            output_database_mini(df_chunk, **kwargs)
    else:
        logger.warning('DataFrame is empty.')


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
        with utils_database.engine_conn('postgre') as conn:
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
        logger.info(f'max_date: {max_date}')
        return max_date


def maximum_date_next(table_name, field_name='date', sql=None):
    max_date = maximum_date(table_name, field_name=field_name, sql=sql)
    next_day = datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=1)
    date_start = next_day.strftime('%Y-%m-%d')
    logger.info(f'date_start: {date_start}')
    return date_start


def feather_file_merge(date_start, date_end):
    date_binary_pair_list = utils_time.date_binary_list(date_start, date_end)
    feather_files = [f'{path}/data/day/{date_binary_pair[0]}.feather' for date_binary_pair in date_binary_pair_list]
    #print(feather_files)
    dfs = [pd.read_feather(file) for file in feather_files if os.path.exists(file)]
    feather_df = pd.concat(dfs, ignore_index=True)
    return feather_df


def find_file(path):
    """
    功能：返回地址下的所有文件
    输出：文件名称列表
    """
    for root, dirs, files in os.walk(path):
        return files


def output_excel(forecast_original, file_path):
    with pd.ExcelWriter(file_path) as writer:
        forecast_original.to_excel(writer, sheet_name='预测表', index=False)


def output_txt(d2):
    with open(f'{path}/data/fight_and_make_trouble.txt', 'w') as f:
        f.write(str(d2.jjd_bh.unique().tolist()))


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


def text_to_text_pd(texts):
    text_pd = pd.DataFrame([range(len(texts)), texts]).T.rename(columns={0: 'xxzjbh', 1: 'text'}).fillna('')
    text_pd.xxzjbh = text_pd.xxzjbh.astype(str)
    return text_pd


def table_in_database(filename):
    with utils_database.engine_conn('postgre') as conn:
        table_exists = pd.read_sql( """SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = '{filename}');""", con=conn.engine).iloc[0, 0]
    if not table_exists:
        return True
    else:
        return False

    #writer = pd.ExcelWriter(f'{path}/data/chongfuzisha_jjd_20220401_20220715.xlsx')
    #data.to_excel(writer, sheet_name='20220401_20220714重复自杀接警单', index=False)
    #writer.save()


def local_matrix(df, field, window = 5, direction='max'):
    return df[field].rolling(window, min_periods=1).max()

# data = pd.DataFrame({'high': np.random.randn(20).cumsum() + 10})
# local_matrix(data,field='high')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-02-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    date_range_df = feather_file_merge(args.date_start, args.date_end)
    print(date_range_df)

# 换文件名os.rename(f'{path}/data/4_property/{file}', f'{path}/data/4_property_horizontal/{file[:9]}horizontal_{file[9:]}')
# 统计数量还原dataframe
# dict_soild_pd.bq_zwm.value_counts().rename_axis('bq_zwm').reset_index(name = '统计_标签总量')
# ws_pos_one_polic['len_ws'] = ws_pos_one_polic.ws.astype(str).str.len()
# ner_pd = ner_pd.astype(str)
# import pinyin
# pinyin.get_initial('中国',delimiter='').upper()
# a[1].astype(str).str.slice(start=-1)datafrmae最后一个字符
# from functools import lru_cache
# {x:consumer_value.get(x,'') for x in label_key_list}
# {x:consumer_value.get(y,'') for (x,y) in zip(risk_key_list,risk_value_list)}
# labels_dic = bq_horizontal[field_list].T[0].to_dict()#对于有N条分类标签。只获取第一条
# 常用短语句
# score_pd.groupby(['ws']).sum()#按照分词累加
# df.A.unique()#唯一值
# df.A.value_counts()#频率2
# person_pd.duplicated('word',keep='first')#判断第一个
# person_pd.drop_duplicates('word', keep='first') 保存第一个
# (datetime.now()+timedelta(days=1)).strftime('%Y%m%d')#明天
# .str.contains
# [x for tem in list_2 for x in tem]#二维转换为一维
# ztk_asj_jqrh_ds['产品@'].dtype=='bool'
# a1=a1.sort_values(by='index_start',ascending=False)排序
# today = datetime.now().strftime("%F")
# yesterday = (datetime.now()+timedelta(days=-1)).strftime('%F')
# .reset_index(drop=True) 序号
# ws_pos_one_polic['len_ws'] = ws_pos_one_polic.ws.astype(str).str.len()

# output1 = pd.merge(output,ztk_asj_jq_ds_jqrh,on='jjd_bh')左右都需要唯一
# path_file = 'C:/Users/admin/Desktop/GeoHzWg.json'
# with open(path_file,'r',encoding='utf-8') as load_f:
#     load_dict = json.load(load_f)
# 键值对：key_value_dict = dict(zip(df['KeyColumn'], df['ValueColumn']))
# df['date_column'] = pd.to_datetime(df['date_column'])
# df['year_column'] = df['date_column'].dt.year
# isinstance(5,int)