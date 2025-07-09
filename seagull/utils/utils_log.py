# -*- coding: utf-8 -*-
"""
@Date: 2024/8/8 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: base_log.py
@Description: 日志

debug: 调试和开发时输出细节
info: 常规信息
success: 成功事件
warning: 潜在问题或意外情况
error: 发生的错误
critical: 最严重的错误，可能导致程序崩溃或终止
"""
import sys
import os
from loguru import logger

# 全局保存 sink id，保证重复调用不会重复添加
_CONSOLE_SINK_ID = None
_FILE_SINK_ID = None

def logger_config_local(
    file_path: str,
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "10 days",
    enable_console: bool = True,
    enable_file: bool = True,
):
    """
    配置 Logger：
    - 单进程：console + file（根据参数开启/关闭）
    - 多进程：主进程调用此函数后，子进程可通过 logger_remove_console() 只保留 File sink

    :param file_path: 日志文件路径（支持格式化，如 "{time:YYYY-MM-DD}.log"）
    :param level: 日志级别
    :param rotation: 文件切割条件
    :param retention: 文件保留时长
    :param enable_console: 是否启用控制台输出
    :param enable_file: 是否启用文件输出
    """
    global _CONSOLE_SINK_ID, _FILE_SINK_ID

    # 第一次调用先清除默认 sink
    if not _CONSOLE_SINK_ID and not _FILE_SINK_ID:
        logger.remove()

    # 添加或更新 console sink
    if enable_console and _CONSOLE_SINK_ID is None:
        _CONSOLE_SINK_ID = logger.add(
            sys.stdout,
            format="{time:YY-MM-DD HH:mm:ss} | {level} | {message} ({file}:{line})",
            level=level,
            enqueue=True,
        )

    # 添加或更新 file sink
    if enable_file and _FILE_SINK_ID is None:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        _FILE_SINK_ID = logger.add(
            file_path,
            format="{time:YY-MM-DD HH:mm:ss} | {level} | {message} ({file}:{line})",
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )

    return logger


def logger_remove_console():
    """
    在子进程中调用，移除 console sink，仅保留 file sink。
    如果 console sink 已经被移除或未添加，什么也不做。
    """
    global _CONSOLE_SINK_ID
    if _CONSOLE_SINK_ID is not None:
        logger.remove(_CONSOLE_SINK_ID)
        _CONSOLE_SINK_ID = None


# def logger_config_local(file_path, level="DEBUG", rotation="10 MB", retention="10 days"):
#     """
#     添加文件日志输出。
#     :param file_path: 日志文件路径
#     :param level: 日志级别
#     :param rotation: 文件轮换条件，例如 "500 MB" 或 "1 week"
#     :param retention: 文件保留时长，例如 "10 days"
#     """
#     logger.add(
#         file_path,
#         format="{time:YY-MM-DD HH:mm:ss} | {level} | {message} ({file}:{line})",
#         level=level,
#         rotation=rotation,
#         retention=retention,
#         enqueue=True,  # 所有写入都经由后台线程，避免多线程直接操作文件
#     )
#     return logger


def logger_config_console():
    # 配置控制台输出
    # logger.add("app.log", format="{time} {level} {message} {file}:{line}", level="DEBUG")
    logger.add(
        # sink=lambda msg: print(msg, end=''),  # 控制台输出
        format="{time:YY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    return logger
