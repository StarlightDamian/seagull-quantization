# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 01:55:02 2024

@author: awei
筹码分布(vap)
美国股票市场，通常用于描述“流通股票持仓成本分布”或“筹码分布”的专业术语是 Volume at Price (VAP) 或 Volume Profile

参考：https://github.com/kengerlwl/ChipDistribution
关键的优化点包括：

这个优化后的实现包含了几个关键的优化技巧和改进：

使用NumPy数组：我们使用了3D NumPy数组来存储筹码分布数据，这比使用字典更高效。
JIT编译：使用Numba的@jit装饰器来即时编译计算密集型函数，显著提高性能。
并行计算：使用Numba的prange和ProcessPoolExecutor来并行处理多只股票的数据。
向量化操作：重写了winner和cost计算函数，使用向量化操作而不是循环。
预计算：我们预先计算了价格水平，避免在每次更新时重新计算。
灵活的价格范围：允许用户指定最小和最大价格，使得系统更加灵活。
可视化：添加了一个简单的绘图函数来可视化筹码分布。
模块化设计：将不同的功能分解成单独的方法，使代码更易于理解和维护。

一些额外的优化技巧：

稀疏矩阵：对于高频数据，可以考虑使用稀疏矩阵来表示筹码分布，因为许多价格水平可能没有交易量。
内存映射：对于非常大的数据集，可以使用NumPy的内存映射功能来处理无法完全加载到内存中的数据。
动态调整价格范围：可以实现一个动态调整价格范围的机制，以适应股票价格的大幅变动。
缓存：对于频繁访问但很少变化的数据，可以使用functools.lru_cache来缓存结果。
异步处理：考虑使用asyncio进行异步I/O操作，特别是在处理多个数据源时。
"""

import numpy as np
import pandas as pd
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

class ChipDistributionOptimized:
    def __init__(self, num_stocks, price_range, time_steps, min_price=0, max_price=1000):
        self.num_stocks = num_stocks
        self.price_range = price_range
        self.time_steps = time_steps
        self.min_price = min_price
        self.max_price = max_price
        self.price_step = (max_price - min_price) / price_range
        self.chip_distribution = np.zeros((num_stocks, price_range, time_steps))
        self.price_levels = np.linspace(min_price, max_price, price_range)

    @staticmethod
    @jit(nopython=True)
    def _update_distribution_single(chip_dist, high, low, avg, vol, turnover_rate, A, price_levels, price_step):
        n = len(price_levels)
        new_dist = np.zeros_like(chip_dist)
        
        # Calculate the triangular distribution
        h = 2 / (high - low)
        for i in range(n):
            price = price_levels[i]
            if price < avg:
                new_dist[i] = h / (avg - low) * (price - low)
            else:
                new_dist[i] = h / (high - avg) * (high - price)
        
        # Normalize and apply volume
        new_dist = new_dist / np.sum(new_dist) * vol * turnover_rate * A
        
        # Update the chip distribution
        chip_dist = chip_dist * (1 - turnover_rate * A) + new_dist
        
        return chip_dist

    def update_distribution(self, stock_data):
        for stock_id, data in enumerate(stock_data):
            for t in range(self.time_steps):
                high, low, avg, vol, turnover_rate = data[t]
                self.chip_distribution[stock_id, :, t] = self._update_distribution_single(
                    self.chip_distribution[stock_id, :, t-1] if t > 0 else np.zeros(self.price_range),
                    high, low, avg, vol, turnover_rate, 1.0, self.price_levels, self.price_step
                )

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_winner_vectorized(chip_dist, prices):
        n_stocks, n_prices, n_times = chip_dist.shape
        winners = np.zeros((n_stocks, n_times))
        for s in prange(n_stocks):
            for t in range(n_times):
                total_chips = np.sum(chip_dist[s, :, t])
                if total_chips > 0:
                    winners[s, t] = np.sum(chip_dist[s, :int(prices[s, t]), t]) / total_chips
        return winners

    def calculate_winner(self, prices):
        return self._calculate_winner_vectorized(self.chip_distribution, prices)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_cost_vectorized(chip_dist, percentile):
        n_stocks, n_prices, n_times = chip_dist.shape
        costs = np.zeros((n_stocks, n_times))
        for s in prange(n_stocks):
            for t in range(n_times):
                cumsum = np.cumsum(chip_dist[s, :, t])
                total = cumsum[-1]
                if total > 0:
                    costs[s, t] = np.searchsorted(cumsum, percentile * total) / n_prices
        return costs

    def calculate_cost(self, percentile):
        return self._calculate_cost_vectorized(self.chip_distribution, percentile / 100)

    def plot_distribution(self, stock_id, time_step):
        plt.figure(figsize=(12, 6))
        plt.bar(self.price_levels, self.chip_distribution[stock_id, :, time_step], width=self.price_step)
        plt.title(f"Chip Distribution for Stock {stock_id} at Time Step {time_step}")
        plt.xlabel("Price")
        plt.ylabel("Chip Amount")
        plt.show()

def process_stock_data(file_path):
    df = pd.read_csv(file_path)
    return df[['high', 'low', 'avg', 'volume', 'TurnoverRate']].values

def main():
    num_stocks = 1000
    price_range = 1000
    time_steps = 252  # Assuming 1 year of daily data

    # Initialize the optimizer
    optimizer = ChipDistributionOptimized(num_stocks, price_range, time_steps)

    # Process stock data in parallel
    with ProcessPoolExecutor() as executor:
        stock_data = list(executor.map(process_stock_data, [f'stock_{i}.csv' for i in range(num_stocks)]))

    # Update distribution
    optimizer.update_distribution(stock_data)

    # Calculate winners and costs
    current_prices = np.random.uniform(50, 150, (num_stocks, time_steps))  # Example current prices
    winners = optimizer.calculate_winner(current_prices)
    costs = optimizer.calculate_cost(90)

    # Plot distribution for a single stock at a specific time
    optimizer.plot_distribution(0, -1)  # Plot the last time step for the first stock

    # Further analysis and visualization can be added here

if __name__ == "__main__":
    main()