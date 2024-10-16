# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:58:26 2024

@author: awei
服务器命令(utils_linux)
hdfs
"""
import os
import socket

def lm_get_cmd_result(cmd):
    """
    功能：获取shell语句的返回值
    输入：shell语句
    """
    content = os.popen(cmd).read()
    return content

def ipv4_address():
    # 创建一个UDP socket连接，不需要实际发送数据，只是为了获取本地主机的IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 连接到一个外部的IP地址 (这里使用了Google的公共DNS服务器IP，但不实际发送数据)
        s.connect(("8.8.8.8", 80))
        # 获取本地主机的IP地址
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "Unable to get IP Address"
    finally:
        s.close()
    
    return ip_address

# 调用函数获取IPv4地址
ipv4 = ipv4_address()
print(f"Current IPv4 Address: {ipv4}")

# cpu核心数
# from multiprocessing import cpu_count
# print(cpu_count())