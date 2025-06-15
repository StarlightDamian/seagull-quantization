# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:59:27 2022

@author: admin
参数管理(base_arguments)
"""
import configparser

from __init__ import path

config = configparser.ConfigParser()
config.read(f'{path}/conf/setting_global.txt')

annual_trading_days = 242  # 一年的交易天数

argparse_user_dict = {
    '达米安': ['科大讯飞', '鸿博股份', '国缆检测','浪潮信息','寒武纪','紫光国微','通达动力','华西股份','三六零','中船科技','璞泰来','中文在线','高新发展','克来机电'],
    '阿妮亚': ['双环传动','宇晶股份','光迅科技','剑桥科技','三联锻造','三柏硕','科华数据','利通电子','德业股份','德新科技' ],
    '张学友': ['引力传媒','赛力斯'],
    '自选':['拓维信息','浪潮信息','科华数据','捷荣技术','清源股份','锦浪科技','金刚光伏','剑桥科技','西典新能','钧达股份','晶澳科技','爱旭股份','赛伍技术','中天科技','人民网','新华网','利通电子','恒润股份','东方精工','亿道信息','神州数码','中电港'],
    }

def get_conf(category, keyword, abnormal_out=''):
    """
    功能：获取/data/setting_global.txt特征配置表，并以一定格式输出
    输入：category:所属主类
        keyword：所需参数的关键词
        abnormal_out：当类型异常时输出什么，或者当什么都没有填的时候的默认值
    输出：conf_str：配置表中对应值，或不满足类型时的默认值
    """
    conf_out = config.get(category, keyword) if config.get(category, keyword) != '' else abnormal_out
    return conf_out



def conf(word, *update):
    """
    功能：调用配置文件的参数
    """
    word = str(word)
    # 连接hive数据库
    if word in ['hive_user', 'hive_password', 'hive_host', 'hive_port', 'hive_database', 'hive_auth']:
        return get_conf('hive_database',word)
    # 连接mysql数据库
    elif word in ['mysql_user', 'mysql_password', 'mysql_host', 'mysql_port', 'mysql_database']:
        return get_conf('mysql_database', word)
    # 连接oracle数据库
    elif word in ['oracle_user', 'oracle_password', 'oracle_host', 'oracle_port', 'oracle_database']:
        return get_conf('oracle_database', word)
    # 连接postgre数据库
    elif word in ['postgre_user', 'postgre_password', 'postgre_host', 'postgre_port', 'postgre_database']:
        return get_conf('postgre_database', word)
    
    # kafka生产者producer
    elif word in ['producer_bootstrap_servers']:
        return get_conf('producer_kafka', word)
    elif word in ['producer_max_request_size', 'producer_acks']:
        return int(get_conf('producer_kafka', word))
    
    # kafka消费者consumer，反馈单
    elif word in ['consumer_fkd_topic', 'consumer_fkd_group_id', 'consumer_fkd_auto_offset_reset', 'consumer_fkd_bootstrap_servers']:
        return get_conf('consumer_kafka_fkd', word)
    
    # kafka消费者consumer，接警单
    elif word in ['consumer_jjd_topic', 'consumer_jjd_group_id', 'consumer_jjd_auto_offset_reset', 'consumer_jjd_bootstrap_servers']:
        return get_conf('consumer_kafka_jjd', word)
    
    # 对接的数据库表名
    elif word in ['table_input', 'table_label_vertical', 'table_property_vertical', 'table_hot_words', 'table_hot_words_details']:
        return get_conf('database_table_name', word)
    
    # 多进程快速刷历史标签
    elif word in ['multiprocessing_date_start', 'multiprocessing_date_end']:
        return get_conf('multiprocessing_label', word)

#base_parameter
if __name__ == '__main__':
    ...
    # args = conf('path')
    # print(args)