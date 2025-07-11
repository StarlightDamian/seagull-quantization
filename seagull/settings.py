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
import sys
sys.path.insert(0, f"{PATH}/venv/Lib/site-packages")

# frequency
FREQ_CODE = {1: "1分钟",
             5: "5分钟",
             15: "15分钟",
             30: "30分钟",
             60: "60分钟",
             101: "日",
             102: "周",
             103: "月"}

# price adjustment
ADJ_CODE = {0: None,  # unadjusted,不复权
            1: "pre",  # forward adjustment,前复权
            2: "post",  # backward adjustment,后复权
            }

ENV = os.environ.get("env_profile", "LOCAL").upper()
if ENV == "LOCAL":
    ANNUAL_TRADING_DAYS = 242  # 一年的交易天数

    USER_SELECTION_DICT = {
        '达米安': ['科大讯飞', '鸿博股份', '国缆检测', '浪潮信息', '寒武纪', '紫光国微', '通达动力', '华西股份',
                  '三六零', '中船科技', '璞泰来', '中文在线', '高新发展', '克来机电'],
        '阿妮亚': ['双环传动', '宇晶股份', '光迅科技', '剑桥科技', '三联锻造', '三柏硕', '科华数据', '利通电子',
                  '德业股份', '德新科技'],
        '张学友': ['引力传媒', '赛力斯'],
        'Yor': ['拓维信息', '浪潮信息', '科华数据', '捷荣技术', '清源股份', '锦浪科技', '金刚光伏', '剑桥科技',
                '西典新能', '钧达股份', '晶澳科技', '爱旭股份', '赛伍技术', '中天科技', '人民网', '新华网', '利通电子',
                '恒润股份', '东方精工', '亿道信息', '神州数码', '中电港'],
    }

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
    raise ValueError(f"Unknown environment type: {ENV}. Please set 'envType' to 'LOCAL', 'STG', or 'PRD'.")


if __name__ == '__main__':
    print(PATH)

