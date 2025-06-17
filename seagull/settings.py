# -*- coding: utf-8 -*-
"""
@Date       : 2025/6/17 11:17
@Author     : Damian
@Email      : zengyuwei1995@163.com
@File       : settings.py
@Description: 
"""
import os

# 上一级目录
parent = os.path.dirname(os.path.dirname(__file__))
PATH = os.path.abspath(parent)

ENV = os.environ.get("env_profile", "LOCAL").upper()
if ENV == "LOCAL":
    # postgres database
    POSTGRES_USER = "postgres"
    POSTGRES_PASSWORD = "zyw8253688"
    POSTGRES_HOST = "127.0.0.1"
    POSTGRES_PORT = 5432
    POSTGRES_DATABASE = "postgres"
    
    # # hive database
    # HIVE_USER =
    # HIVE_PASSWORD =
    # HIVE_HOST =
    # HIVE_PORT =
    # HIVE_DATABASE =
    # HIVE_AUTH =
    #
    # # mysql database]
    # MYSQL_USER =
    # MYSQL_PASSWORD =
    # MYSQL_HOST =
    # MYSQL_PORT =
    # MYSQL_DATABASE =
    #
    # # oracle database
    # ORACLE_USER =
    # ORACLE_PASSWORD =
    # ORACLE_HOST =
    # ORACLE_PORT =
    # ORACLE_DATABASE =

elif ENV == "STG":
    ...
elif ENV == "PRD":
    ...
else:
    ...

if __name__ == '__main__':
    print(PATH)

