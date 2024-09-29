# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:09:23 2024

@author: awei
日志(base_log)
debug: 调试和开发时输出细节
info: 常规信息
success: 成功事件
warning: 潜在问题或意外情况
error: 发生的错误
critical: 最严重的错误，可能导致程序崩溃或终止
"""
from loguru import logger

def logger_config_local(file_path, level="DEBUG", rotation="10 MB", retention="10 days"):
    """
    添加文件日志输出。
    :param file_path: 日志文件路径
    :param level: 日志级别
    :param rotation: 文件轮换条件，例如 "500 MB" 或 "1 week"
    :param retention: 文件保留时长，例如 "10 days"
    """
    logger.add(
        file_path,
        format="{time:YY-MM-DD HH:mm:ss} | {level} | {message} ({file}:{line})",
        level=level,
        rotation=rotation,
        retention=retention
    )
    return logger

def logger_config_base():
    # 配置控制台输出
    # logger.add("app.log", format="{time} {level} {message} {file}:{line}", level="DEBUG")
    logger.add(
        sink=lambda msg: print(msg, end=''),  # 控制台输出
        format="{time:YY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    return logger
