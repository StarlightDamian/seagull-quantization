# -*- coding: utf-8 -*-
"""
@Date: 2025/7/8 11:36
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_ikun.py
@Description: 
"""
import os
import subprocess
import sys

def load_msvc_env():
    # 根据实际安装路径修改
    vs_path = r"D:\Program Files\Microsoft Visual Studio\2022\Community"
    vcvars = os.path.join(
        vs_path,
        r"VC\Auxiliary\Build\vcvarsall.bat"
    )
    arch = "x64"  # 或 "x86"
    # 调用批处理，并捕获它对环境变量所做的更改
    completed = subprocess.run(
        f'"{vcvars}" {arch} & set',
        shell=True,
        stdout=subprocess.PIPE,
        text=True
    )
    # 解析输出，把每一行 VAR=VALUE 都放到 os.environ
    for line in completed.stdout.splitlines():
        key, sep, val = line.partition("=")
        if sep:
            os.environ[key] = val

from seagull.settings import PATH
from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined import Alpha101
from KunQuant.runner import KunRunner as kr

builder = Builder()
with builder:
    vclose = Input("close")
    low = Input("low")
    high = Input("high")
    vopen = Input("open")
    amount = Input("amount")
    vol = Input("volume")
    all_data = Alpha101.AllData(low=low,high=high,close=vclose,open=vopen, amount=amount, volume=vol)
    Output(Alpha101.alpha001(all_data), "alpha001")
    Output(Alpha101.alpha002(all_data), "alpha002")
f = Function(builder.ops)
# 先加载 MSVC / Windows SDK 环境
load_msvc_env()
# 下面正常 import cfake 并编译
# from cfake import compileit, CppCompilerConfig, KunCompilerConfig

lib = cfake.compileit(
    [("alpha101", f, KunCompilerConfig(input_layout="TS",
                                       output_layout="TS"))],
    "out_first_lib",
    cfake.CppCompilerConfig()
)

modu = lib.getModule("alpha101")

import pandas as pd
from seagull.utils import utils_database

import numpy as np
import pandas as pd
path = 'D:/03_software_engineering/05_github/seagull'
raw_df = pd.read_feather(f'{path}/_file/das_wide_incr_train_mini.feather')
#raw_df = raw_df[raw_df.full_code=='000001']
#raw_df = raw_df[raw_df.full_code=='000001']
raw_df = raw_df[(raw_df.date>='2020-06-30')]#2011-06-30
#['date', 'full_code', 'open', 'close', 'high', 'low', 'volume','turnover', 'amount']
df = []
for full_code in raw_df.full_code.unique():
    df1 = raw_df[raw_df.full_code==full_code]
    del df1['full_code']
    df1.set_index('date', drop=True, inplace=True)
    df.append(df1)
import numpy as np

# [features, stocks, time]
cols = df[0].columns.values # array(['date', 'full_code', 'open', 'close', 'high', 'low', 'volume','turnover', 'amount'], dtype=object)
col2idx = dict(zip(cols, range(len(cols))))
#collected = np.empty((len(col2idx), len(df.full_code.unique()), df.shape[0]), dtype="float32")
collected = np.empty((len(col2idx), len(raw_df.full_code.unique()), len(df[0])), dtype="float32")
for stockidx, data in enumerate(df):
    for colname, colidx in col2idx.items():
        mat = data[colname].to_numpy()
        collected[colidx, stockidx, :] = mat

# [features, stocks, time] => [features, time, stocks]
transposed = collected.transpose((0, 2, 1))
transposed = np.ascontiguousarray(transposed)

input_dict = dict()
for colname, colidx in col2idx.items():
    input_dict[colname] = transposed[colidx]

# using 4 threads
num_time = len(df[0])
executor = kr.createMultiThreadExecutor(4)
out = kr.runGraph(executor, modu, input_dict, 0, num_time)
print("Result of alpha101", out["alpha002"])
print("Shape of alpha101", out["alpha002"].shape)
# with utils_database.engine_conn('POSTGRES') as conn:
#    trading_day_df = pd.read_sql("select date from dwd_base_full_trading_day where trade_status=1", con=conn.engine)

# =============================================================================
# all_alpha = [alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha007, alpha008, alpha009, alpha010,
#     alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020, alpha021,
#     alpha022, alpha023, alpha024, alpha025, alpha026, alpha027, alpha028, alpha029, alpha030, alpha031, alpha032,
#     alpha033, alpha034, alpha035, alpha036, alpha037, alpha038, alpha039, alpha040, alpha041, alpha042, alpha043,
#     alpha044, alpha045, alpha046, alpha047, alpha049, alpha050, alpha051, alpha052, alpha053, alpha054, alpha055,
#     alpha057, alpha060, alpha061, alpha062, alpha064, alpha065, alpha066, alpha068, alpha071, alpha072, alpha073,
#     alpha074, alpha075, alpha077, alpha078, alpha081, alpha083, alpha084, alpha085, alpha086, alpha088, alpha092,
#     alpha094, alpha095, alpha096, alpha098, alpha099, alpha101
#     ]
# =============================================================================