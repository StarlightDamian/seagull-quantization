# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:53:45 2022

@author: admin
连接数据库(utils_database)

数据库连接模块考虑情况：
1.数据库连接池，连接数量
2.engine连接，普通连接，他们读写数据库表的写法会有差异
3.考虑windows和linux连接差异，PooledDB包在不同系统下的读取方式不同。连接hive在windows环境需要impala
4.有时候项目只需要部分类型的包支持指定数据库，如只安装hive和postgre相应的包
5.不同类型连接的参数不一致，比如hive有'auth'、'auth_mechanism'
6.支持windows和Linux连接同一类数据库，但是host：post不一致。windows测试，Linux正式

schema
"""
import os
from urllib import parse
from sqlalchemy import create_engine, inspect
from datetime import datetime, timedelta
import psycopg2

import logger
import pandas as pd

from __init__ import path
from utils import utils_arguments as arg


class DatabaseConnection:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def __enter__(self):
        self.conn = self.engine.connect()
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()

    def _generate_create_table_sql(self, data_df, table_name):
        # 生成 CREATE TABLE 语句，基于 DataFrame 的列名和数据类型
        columns = []
        for col, dtype in data_df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = 'FLOAT'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = 'TIMESTAMP'
            else:
                sql_type = 'TEXT'
            columns.append(f"{col} {sql_type}")
        
        columns_sql = ", ".join(columns)
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        return create_table_sql
    
    def large_data_output_database(self, data_df, table_name):
        # 1. 添加时间戳
        data_df['insert_timestamp'] = datetime.now().strftime("%F %T")
        
        # 2. 动态创建表的 SQL 语句
        create_table_sql = self._generate_create_table_sql(data_df, table_name)
        with self.conn.engine.connect() as connection:
            connection.execute(create_table_sql)
        
        # 3. 将 DataFrame 写入 CSV 文件
        csv_file = f'{path}/cache/{table_name}.csv'
        data_df.to_csv(csv_file, index=False, sep='\t')

        # 4. 使用 COPY 命令将数据导入 PostgreSQL
        with self.conn.engine.raw_connection() as raw_conn:
            cursor = raw_conn.cursor()
            cursor.execute(f"""
                COPY {table_name} FROM '{csv_file}' DELIMITER '\t' CSV HEADER;
            """)
            raw_conn.commit()

        # 5. 删除临时 CSV 文件
        os.remove(csv_file)

        
def database_maximum_date(table_name, field_name):
    try:
        with engine_conn('postgre') as conn:
            max_date_df = pd.read_sql(f"SELECT max({field_name}) FROM {table_name}", con=conn.engine)
        max_date = max_date_df.values[0][0]
        logger.info(f'max_date: {max_date}')
    except:
        logger.error('Exception in querying database maximum date')
        max_date = '1990-01-01'
    finally:
        next_day = datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=1)
        date_start = next_day.strftime('%Y-%m-%d')
        logger.info(f'date_start: {date_start}')
        return date_start

        
def psycopg2_conn():
    # 创建数据库连接
    conn = psycopg2.connect(
        dbname='postgres', 
        user='postgres', 
        password='zyw8253688',
        host='127.0.0.1', 
        port='5432'
    )
    return conn

def engine_url(type_database):
    user = arg.conf(f'{type_database}_user')
    password = arg.conf(f'{type_database}_password')
    password = parse.quote_plus(str(password))  # 处理密码中带有@，被create_engine误分割导致的BUG
    host = arg.conf(f'{type_database}_host')
    port = arg.conf(f'{type_database}_port')
    database = arg.conf(f'{type_database}_database')
    database_dict = {'hive': 'hive', 'postgre': 'postgresql', 'oracle': 'oracle', 'mysql': 'mysql+pymysql'}
    database_name = database_dict.get(f"{type_database}")
    user_password_host_port_database_str = f"{user}:{password}@{host}:{port}/{database}"

    if type_database == 'hive':
        auth = arg.conf('hive_auth')
        db_url = f"{database_name}://{user_password_host_port_database_str}?auth={auth}"
    elif type_database in ['postgre', 'oracle', 'mysql']:
        db_url = f"{database_name}://{user_password_host_port_database_str}"
    return  db_url

def table_exists(tablename):
    db_url = engine_url('postgre')
    engine = create_engine(db_url)
    inspector = inspect(engine)
    return inspector.has_table(tablename)

def engine_conn(type_database):
    """
    功能：连接数据库
    备注：输出至数据库：to_csv()  if_exists:['append','replace','fail']#追加、删除原表后新增、啥都不干抛出一个 ValueError
    """
    #print(f"当前数据库：{type_database}")
    user = arg.conf(f'{type_database}_user')
    password = arg.conf(f'{type_database}_password')
    password = parse.quote_plus(str(password))  # 处理密码中带有@，被create_engine误分割导致的BUG
    host = arg.conf(f'{type_database}_host')
    port = arg.conf(f'{type_database}_port')
    database = arg.conf(f'{type_database}_database')
    database_dict = {'hive': 'hive', 'postgre': 'postgresql', 'oracle': 'oracle', 'mysql': 'mysql+pymysql'}
    database_name = database_dict.get(f"{type_database}")
    user_password_host_port_database_str = f"{user}:{password}@{host}:{port}/{database}"

    if type_database == 'hive':
        auth = arg.conf('hive_auth')
        db_url = f"{database_name}://{user_password_host_port_database_str}?auth={auth}"
    elif type_database in ['postgre', 'oracle', 'mysql']:
        db_url = f"{database_name}://{user_password_host_port_database_str}"
    # print(db_url)
    return DatabaseConnection(db_url)

if __name__ == '__main__':
    # print(path)
    
    # Data exploration
    with engine_conn('postgre') as conn:
        table_name = 'ods_ohlc_incr_efinance_stock_bj_daily'
        count_field = '日期'
        #data = pd.read_sql("SELECT * FROM history_a_stock_k_data limit 10", con=conn.engine)  # Use conn_pg.engine
        data = pd.read_sql(f"SELECT {count_field}, COUNT(*) AS data_count FROM {table_name} GROUP BY {count_field} ORDER BY {count_field}", con=conn.engine)
        print(data)
    
    print('The amount of data:', data.data_count.sum())

    
        
        
    #data.to_csv(f'{path}/data/history_a_stock_k_data_count_date.csv')
    
    #from base import base_utils
    #data['primaryKey'] = (data['date']+data['code']).apply(base_utils.md5_str) # md5（日期、时间、代码）
    #from sqlalchemy import Float, Numeric, String
# =============================================================================
#     data.to_sql('history_a_stock_k_data', con=conn.engine, index=False, if_exists='replace',
#                             dtype={
#                                 'primaryKey': String,
#                                 'date': String,
#                                 'code': String,
#                                 'code_name': String,
#                                 'open': Float,
#                                 'high': Float,
#                                 'low': Float,
#                                 'close': Float,
#                                 'preclose': Float,
#                                 'volume': Numeric,
#                                 'amount': Numeric,
#                                 'adjustflag': String,
#                                 'turn': Float,
#                                 'tradestatus': String,
#                                 'pctChg': Float,
#                                 'peTTM': Float,
#                                 'psTTM': Float,
#                                 'pcfNcfTTM': Float,
#                                 'pbMRQ': Float,
#                                 'isST': String,
#                             })
# =============================================================================
    # conn_pg = engine_conn('postgre')
    # data = pd.read_sql("SELECT * FROM warning_hot_word_mx limit 10",con=conn_pg)
    
    #writer = pd.ExcelWriter(f'{path}/data/chongfuzisha_jjd_20220401_20220715.xlsx')
    #data.to_excel(writer, sheet_name='20220401_20220714重复自杀接警单', index=False)
    #writer.save()
