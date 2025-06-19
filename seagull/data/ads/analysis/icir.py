# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:35:05 2025

@author: awei
icir3
"""
import pandas as pd
import numpy as np
import alphalens as al
import matplotlib.pyplot as plt


def calculate_icir(factor_data, price_data):
    factor_df = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data,  # 因子数据
        prices=price_data,  # 股票价格数据
        periods=(1, 5, 10, 22),  # 计算未来1日、5日和10日的收益率
        )
    
    # 计算因子与未来收益的IC（信息系数）
    ic = al.performance.factor_information_coefficient(factor_df)
    ir = ic.mean() / ic.std()
    return ic, ir

if __name__ == "__main__":
    # 随机生成股票价格数据（价格数据假设从2015年1月1日至2016年12月31日）
    dates = pd.date_range('2015-01-01', '2016-12-31', freq='B')  # 只选择工作日
    assets  = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA']  # 假设有5只股票
    price_data = pd.DataFrame(np.random.randn(len(dates),
                              len(assets)) * 5 + 100,
                              index=dates,
                              columns=assets)
    price_data = price_data.abs()  # 价格必须是正数
    price_data.index.set_names(['date'], inplace=True)
    
    # 假设因子数据：每天计算每只股票的因子值（这里使用随机数作为因子数据）
    factor_data = pd.DataFrame({
    "factor": np.random.randn(len(dates) * len(assets)),
}, index=pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]))
    
    ic, ir = calculate_icir(factor_data, price_data)
    
    
    # 因子衰减分析
    factor_returns = al.performance.factor_returns(
        factor_data,
        demeaned=True,
        group_adjust=False,
    )
    
# =============================================================================
#     # 因子收益分析
#     mean_return, std_error_return = al.performance.mean_return_by_quantile(
#         factor_data,
#         by_date=True,
#         by_group=False,
#         demeaned=True,
#         group_adjust=False,
#     )
# =============================================================================
# =============================================================================
#     # 因子收益图
#     al.plotting.plot_factor_returns(factor_returns)
#     
#     # 信息系数（IC）图
#     al.plotting.plot_ic_heatmap(ic)
#     
#     # 可视化分析
#     al.plotting.create_full_tear_sheet(
#         factor_data,
#         price_data,
#         long_short=True,
#         group_neutral=False,
#         by_group=False
#     )
# 
# =============================================================================
