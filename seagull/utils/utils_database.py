# -*- coding: utf-8 -*-
"""
@Date: 2022/5/18 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_database.py
@Description: 连接数据库

数据库连接模块考虑情况：
1.数据库连接池，连接数量
2.engine连接，普通连接，他们读写数据库表的写法会有差异
3.考虑windows和linux连接差异，PooledDB包在不同系统下的读取方式不同。连接hive在windows环境需要impala
4.有时候项目只需要部分类型的包支持指定数据库，如只安装hive和postgre相应的包
5.不同类型连接的参数不一致，比如hive有'auth'、'auth_mechanism'
6.支持windows和Linux连接同一类数据库，但是host：post不一致。windows测试，Linux正式
"""
import os
from urllib import parse
from datetime import datetime, timedelta

from sqlalchemy import create_engine, inspect
import psycopg2
import logger
import pandas as pd

from seagull import settings


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
        with engine_conn('POSTGRES') as conn:
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
    try:
        user = getattr(settings, "POSTGRES_USER")
        password = getattr(settings, "POSTGRES_PASSWORD")
        password = parse.quote_plus(str(password))  # 处理密码中带有@，被create_engine误分割导致的BUG
        host = getattr(settings, "POSTGRES_HOST")
        port = getattr(settings, "POSTGRES_PORT")
        database = getattr(settings, "POSTGRES_DATABASE")
    except AttributeError as e:
        raise RuntimeError(f"No such database config: {e}")

    # 创建数据库连接
    conn = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn


def engine_url(database_type):
    try:
        user = getattr(settings, f"{database_type}_USER")
        password = getattr(settings, f"{database_type}_PASSWORD")
        password = parse.quote_plus(str(password))  # 处理密码中带有@，被create_engine误分割导致的BUG
        host = getattr(settings, f"{database_type}_HOST")
        port = getattr(settings, f"{database_type}_PORT")
        database = getattr(settings, f"{database_type}_DATABASE")
    except AttributeError as e:
        raise RuntimeError(f"No such database config: {e}")

    database_dict = {'POSTGRES': 'postgresql',
                     'ORACLE': 'oracle',
                     'MYSQL': 'mysql+pymysql'}
    database_name = database_dict.get(f"{database_type}")
    user_password_host_port_database_str = f"{user}:{password}@{host}:{port}/{database}"

    if database_type == 'HIVE':
        auth = getattr(config, "HIVE_AUTH")
        db_url = f"hive://{user}:{password}@{host}:{port}/{database}?auth={auth}"
    elif database_type in ['POSTGRES', 'ORACLE', 'MYSQL']:
        db_url = f"{database_name}://{user_password_host_port_database_str}"
    return db_url


def table_exists(tablename):
    db_url = engine_url('POSTGRES')
    engine = create_engine(db_url)
    inspector = inspect(engine)
    return inspector.has_table(tablename)


def engine_conn(database_type):
    """
    功能：连接数据库
    备注：输出至数据库：to_csv()  if_exists:['append','replace','fail']#追加、删除原表后新增、啥都不干抛出一个 ValueError
    """
    db_url = engine_url(database_type)
    return DatabaseConnection(db_url)


if __name__ == '__main__':
    print(settings.PATH)
    with engine_conn('POSTGRES') as pg_conn:
        data = pd.read_sql(f"SELECT * FROM ods_ohlc_incr_efinance_stock_minute_1 limit 10", con=pg_conn.engine)
        print(data)
