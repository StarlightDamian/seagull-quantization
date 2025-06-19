# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:17:03 2024

@author: awei
vap5
"""
import os
import argparse

import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

class ChipDistributionAnalyzer:
    def __init__(self, price_range=100, min_price=0, max_price=1000, period=730):
        self.price_range = price_range
        self.min_price = min_price
        self.max_price = max_price
        self.price_levels = np.linspace(min_price, max_price, price_range)
        self.period = period  # 滚动窗口的天数（两年）

    def calculate_chip_peak(self, group):
        # 筛选有效的价格区间并计算筹码峰
        group = group.copy()
        group['chip_peak'] = np.where(
            (group['high'] > group['low']),
            group[['low', 'high']].mean(axis=1),  # 使用高低均值作为筹码峰
            group['avg_price']  # 如果high和low相同，取平均价格
        )
        return group

    def calculate_distribution_ranges(self, group):
        # 滚动窗口计算盈亏分布
        result = group.copy()
        
        # 获取盈利分布的价格区间
        profit_bins = np.linspace(-100, 100, 21)  # 20个区间
        bin_labels = [f"[{int(profit_bins[i])}, {int(profit_bins[i+1])}]" for i in range(len(profit_bins) - 1)]
        
        # 初始化盈亏列
        for label in bin_labels:
            result[label] = 0

        # 两年滚动窗口
        for i in range(self.period, len(result)):
            window = result.iloc[i - self.period:i]
            current_price = result.iloc[i]['avg_price']

            # 计算盈亏比例分布
            price_ratios = (window['avg_price'] - current_price) / current_price * 100
            hist, _ = np.histogram(price_ratios, bins=profit_bins)

            # 将分布写入DataFrame
            for j, label in enumerate(bin_labels):
                result.at[result.index[i], label] = hist[j]

        return result

    def analyze_chip_distribution(self, stock_data):
        # 分组并计算筹码峰和分布
        stock_data = stock_data.groupby('full_code').apply(self.calculate_chip_peak).reset_index(drop=True)
        #stock_data = stock_data.groupby('full_code').apply(self.calculate_distribution_ranges)
        
        return stock_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--date_start', type=str, default='2014-01-01', help='Start time for training')
    parser.add_argument('--date_start', type=str, default='2020-01-20', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='end time of training')
    args = parser.parse_args()
    
    logger.info(f"""
    task: train_2_stock_price_high
    date_start: {args.date_start}
    date_end: {args.date_end}""")
    
    # dataset
    with utils_database.engine_conn("POSTGRES") as conn:
        stock_daily_df = pd.read_sql(f"""
        SELECT 
            primary_key,
            full_code,
            date,
            high,
            low,
            volume,
            value_traded,
            avg_price,
            turnover
        FROM
            dwd_freq_incr_stock_daily
        WHERE 
            date BETWEEN '{args.date_start}' AND '{args.date_end}'
            AND full_code in ('601127.sh', '601919.sh','000560.sz')
                                     """, con=conn.engine)
    stock_daily_df.drop_duplicates('primary_key',keep='first', inplace=True)
    # 假设已经加载stock_data并准备好current_prices
    analyzer = ChipDistributionAnalyzer()
    chip_dist_df = analyzer.analyze_chip_distribution(stock_daily_df)
    chip_dist_df.to_csv(f'{PATH}/_file/chip_dist.csv', index=False)
    #results = analyzer.analyze_chip_distribution(chip_dist_df, current_prices)
    #print(results)
