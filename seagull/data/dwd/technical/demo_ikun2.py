# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 13:41
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_ikun2.py
@Description: 
"""
import os
import sys
import time
import threading
import subprocess
from datetime import datetime

from seagull.settings import PATH


def load_msvc_env(vs_path=r"D:\Program Files\Microsoft Visual Studio\2022\Community", arch="x64"):
    # MSVC 环境加载（仅限 Windows，若不需可移除）
    vcvars = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvarsall.bat")
    cmd = f'"{vcvars}" {arch} & set'
    completed = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    for line in completed.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v

# KunQuant 引擎导入
from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig

from KunQuant.Stage import Function
from KunQuant.runner import KunRunner as kr


#import sys
#sys.path.insert(0, "D:/03_software_engineering/07_open_source_important/KunQuant-main/KunQuant")
from KunQuant.predefined import Alpha101
# 自动收集 Alpha101 中所有以 alpha 开头的方法
import inspect
alpha_funcs = {
    name: func
    for name, func in inspect.getmembers(Alpha101, inspect.isfunction)
    if name.startswith("alpha")
}

# 构造计算图
from KunQuant.Op import Builder, Input, Output
builder = Builder()
with builder:
    # 定义输入字段
    inputs = {field: Input(field) for field in ('open', 'high', 'low', 'close', 'volume', 'amount')}
    # 聚合到 AllData
    all_data = Alpha101.AllData(**inputs)
    # 按函数名批量输出
    for name, func in alpha_funcs.items():
        Output(func(all_data), name)

# 编译
f = Function(builder.ops)
load_msvc_env()  # 如果非 Windows，可注释
lib = cfake.compileit(
    [("alpha101", f, KunCompilerConfig(input_layout="TS", output_layout="TS"))],
    "out_first_lib",
    cfake.CppCompilerConfig()
)
modu = lib.getModule("alpha101")

# 数据准备
import pandas as pd
import numpy as np
from seagull.utils import utils_database
path = 'D:/03_software_engineering/05_github/seagull'
raw = pd.read_feather(f"{path}/_file/das_wide_incr_train_mini.feather")
raw = raw[raw.date >= '2020-06-30']

# 构建 3D 数组 [fields, time, stocks]
groups = raw.groupby('full_code')
fields = [col for col in raw.columns if col not in ('date', 'full_code')]
t0 = groups.first().shape[0]
time_len = groups.size().iloc[0]
arr = np.empty((len(fields), time_len, t0), dtype='float32')

for idx, (code, df_stock) in enumerate(groups):
    df_stock = df_stock.set_index('date').loc[:, fields]
    arr[:, :, idx] = df_stock[fields].values.T

# 构建输入字典
dict_in = {f: arr[i] for i, f in enumerate(fields)}

# 运行
executor = kr.createMultiThreadExecutor(4)
out = kr.runGraph(executor, modu, dict_in, 0, time_len)

# 输出示例
for name in sorted(alpha_funcs):
    print(f"{name}: shape={out[name].shape}")
