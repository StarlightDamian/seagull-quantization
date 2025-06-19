# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:38:09 2024

@author: awei
"""

# 导入所需模块
import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.report import analysis_model, analysis_position

# 初始化qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)

# 数据处理模块示例
def data_handler_example():
    handler = Alpha158(start_time='2010-01-01', end_time='2020-12-31', fit_start_time='2010-01-01', fit_end_time='2020-12-31')
    dataset = handler.get_dataset()
    print(dataset)

# 模型训练模块示例
def model_training_example():
    model = LGBModel()
    dataset = Alpha158(start_time='2010-01-01', end_time='2020-12-31', fit_start_time='2010-01-01', fit_end_time='2020-12-31')
    model.fit(dataset)
    prediction = model.predict(dataset)
    print(prediction)

# 策略模块示例
def strategy_example():
    strategy = TopkDropoutStrategy(model=LGBModel(), dataset=Alpha158(), topk=50, n_drop=5)
    strategy.prepare_data()
    strategy.trade()

# 回测模块示例
def backtest_example():
    strategy = TopkDropoutStrategy(model=LGBModel(), dataset=Alpha158(), topk=50, n_drop=5)
    report_normal, positions_normal = backtest_daily(strategy)
    analysis_model(report_normal)
    analysis_position(positions_normal)

# 运行示例
if __name__ == "__main__":
    print("数据处理示例:")
    data_handler_example()
    print("\n模型训练示例:")
    model_training_example()
    print("\n策略示例:")
    strategy_example()
    print("\n回测示例:")
    backtest_example()