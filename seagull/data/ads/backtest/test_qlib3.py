# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:51:34 2024

@author: awei
"""
import sys
sys.path.append('E:/03_software_engineering/open_source_important/qlib-main/qlib')
# 导入必要的库
import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, backtest_daily_fm
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import exists_qlib_data, get_module_by_module_path, flatten_dict
import pandas as pd

# 初始化QLib
def initialize_qlib():
    provider_uri = "~/.qlib/qlib_data/cn_data"  # 确保您已经下载了相应的数据集
    qlib.init(provider_uri=provider_uri, region=REG_CN)

# 数据准备函数
def prepare_data(start_time, end_time, instruments):
    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": end_time,
        "instruments": instruments,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
        "filter_pipe": [
            {"class": "MCFilter"},
            {"class": "DropnaFilter"},
            {"class": "FilterByInstrument", "kwargs": {"field_group": "feature"}}
        ]
    }
    return data_handler_config

# 模型配置函数
def get_model_config():
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    }
    return model_config

# 任务配置函数
def get_task_config(data_handler_config, model_config):
    task_config = {
        "model": model_config,
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }
    return task_config

# 策略配置函数
def get_strategy_config():
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.strategy",
        "kwargs": {
            "topk": 50,
            "n_drop": 5,
        },
    }
    return strategy_config

# 回测配置函数
def get_backtest_config():
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }
    return backtest_config

# 主函数
def main():
    # 初始化QLib
    initialize_qlib()
    
    # 准备数据
    instruments = "csi300"  # 或者使用您自定义的股票池
    data_handler_config = prepare_data("2008-01-01", "2020-08-01", instruments)
    
    # 获取模型配置
    model_config = get_model_config()
    
    # 获取任务配置
    task_config = get_task_config(data_handler_config, model_config)
    
    # 创建并执行任务
    task_obj = init_instance_by_config(task_config)
    task_obj.fit()
    
    # 预测
    pred_obj = task_obj.predict(segment="test")
    
    # 保存预测结果
    recorder = R.get_recorder()
    sr = SignalRecord(model=task_obj, dataset=task_obj.dataset, record=recorder)
    sr.produce(pred_obj, "test")
    
    # 获取策略配置
    strategy_config = get_strategy_config()
    
    # 获取回测配置
    backtest_config = get_backtest_config()
    
    # 执行回测
    backtest_result = backtest_daily_fm(
        pred_obj,
        strategy_config,
        backtest_config,
    )
    
    # 保存回测结果
    par = PortAnaRecord(
        recorder,
        backtest_config["start_time"],
        backtest_config["end_time"]
    )
    par.generate(backtest_result)
    
    # 输出回测结果
    print(backtest_result)

if __name__ == "__main__":
    main()