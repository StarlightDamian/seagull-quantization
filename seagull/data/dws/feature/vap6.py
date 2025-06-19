# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:20:10 2024

@author: awei
vap6
"""

import os
import argparse

import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class ChipDistributionOptimized:
    def __init__(self):
        self.chip_df = pd.DataFrame()  # 保存每个价位的筹码分布

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath, parse_dates=['date'])

    def calculate_chip_distribution(self, minD=0.01, AC=1):
        """
        计算筹码分布，按每个价位分布并按成交量衰减
        """
        price_ranges = []
        volume_distributions = []

        # 计算每日筹码分布并记录所有价位范围
        for _, row in self.data.iterrows():
            high, low, vol, turnover_rate = row['high'], row['low'], row['volume'], row['TurnoverRate'] / 100
            price_range = np.arange(low, high, minD)
            each_volume = vol / len(price_range)

            # 记录每日的价位和对应成交量
            price_ranges.append(pd.Series(each_volume, index=price_range))
            volume_distributions.append(turnover_rate * AC)

        # 将筹码分布转换为DataFrame，并用Pandas的concat处理所有日期的合并
        self.chip_df = pd.concat(price_ranges, axis=1).fillna(0)
        self.chip_df.columns = self.data['date']

        # 按成交量衰减历史筹码
        decay_factors = pd.Series(volume_distributions, index=self.data['date'])
        decay_factors = (1 - decay_factors).cumprod()
        self.chip_df = self.chip_df.mul(decay_factors, axis=1)

    def calculate_profit_range(self):
        """
        根据收益范围生成分段列 [-100%, 100%] 分为20段
        """
        range_bins = np.linspace(-1, 1, 21)
        labels = [f"{int(b * 100)}-{int(range_bins[i + 1] * 100)}%" for i, b in enumerate(range_bins[:-1])]
        
        # 使用Pandas的cut函数直接计算收益范围
        self.data['profit_bins'] = pd.cut(self.data['close'].pct_change(), bins=range_bins, labels=labels, include_lowest=True)

    def calculate_winner(self):
        """
        计算每个价位的获利比例
        """
        # 使用矢量化计算 close price 与chip_df中价格位置的比较
        close_prices = self.data['close'].values
        win_ratios = ((self.chip_df.index.values[:, None] < close_prices).astype(int) * self.chip_df).sum(axis=0)
        total_volume = self.chip_df.sum(axis=0)
        self.data['winner'] = win_ratios / total_volume  # 获利盘比例

    def get_data(self):
        return self.data
    #def run_all_calculations(self, filepath, minD=0.01, AC=1):

        # a.cost(90) #成本分布
        #return self.data[['date', 'winner', 'profit_bins']]

if __name__ == "__main__":
    chip=ChipDistributionOptimized()
    #data = a.run_all_calculations(f'{PATH}/_file/test.csv')
    chip.load_data(f'{PATH}/_file/test.csv')
    chip.calculate_chip_distribution(minD=0.01, AC=1)
    chip.calculate_profit_range()
    chip.calculate_winner()
a.cost(90) #成本分布
    data = chip.get_data()

    #a.lwinner()