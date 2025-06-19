# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:19:59 2024

@author: awei
1.update_distribution：使用NumPy的向量化函数来更新芯片分布，同时避免逐个时间步的显式循环。
2.calculate_winner 和 calculate_cost：通过向量化计算，在一个步骤中同时计算所有股票的胜率和成本。
3.stock_data：这是通过Pandas生成的数据框，包含每只股票的历史高低点和成交量数据。

流动性：正常是1-3年
高流动性是6月-1年
低流动性是2-3年
"""
import pandas as pd
import numpy as np

class ChipDistributionVectorized:
    def __init__(self, num_stocks, price_range, time_steps, min_price=0, max_price=1000):
        self.num_stocks = num_stocks
        self.price_range = price_range
        self.time_steps = time_steps
        self.min_price = min_price
        self.max_price = max_price
        self.price_step = (max_price - min_price) / price_range
        self.chip_distribution = np.zeros((num_stocks, price_range, time_steps))
        self.price_levels = np.linspace(min_price, max_price, price_range)

    def update_distribution(self, df):
        # Group by stock and time_step, then apply vectorized calculations
        for stock_id in df['stock_id'].unique():
            stock_data = df[df['stock_id'] == stock_id].copy()
            for t in range(self.time_steps):
                high, low, avg, vol, turnover_rate = stock_data.iloc[t][['high', 'low', 'avg', 'volume', 'TurnoverRate']].values
                if t > 0:
                    self.chip_distribution[stock_id, :, t] = self._update_distribution_single(
                        self.chip_distribution[stock_id, :, t-1],
                        high, low, avg, vol, turnover_rate, 1.0, self.price_levels, self.price_step
                    )
                else:
                    self.chip_distribution[stock_id, :, t] = self._update_distribution_single(
                        np.zeros(self.price_range), high, low, avg, vol, turnover_rate, 1.0, self.price_levels, self.price_step
                    )

    def _update_distribution_single(self, chip_dist, high, low, avg, vol, turnover_rate, A, price_levels, price_step):
        n = len(price_levels)
        new_dist = np.zeros_like(chip_dist)
        
        # Calculate the triangular distribution
        h = 2 / (high - low)
        new_dist = np.where(price_levels < avg,
                            h / (avg - low) * (price_levels - low),
                            h / (high - avg) * (high - price_levels))
        
        # Normalize and apply volume
        new_dist = new_dist / np.sum(new_dist) * vol * turnover_rate * A
        
        # Update the chip distribution
        chip_dist = chip_dist * (1 - turnover_rate * A) + new_dist
        return chip_dist
    
    def calculate_chip_range(costs, price_levels, percentile=90):
        """
        Calculate the price range for the given percentile of chips.
        计算集中度
        """
        chip_range = []
        for stock_id in range(costs.shape[0]):
            for t in range(costs.shape[1]):
                low_index = int(np.percentile(np.cumsum(costs[stock_id, :, t]), (100 - percentile)))
                high_index = int(np.percentile(np.cumsum(costs[stock_id, :, t]), percentile))
                chip_range.append([price_levels[low_index], price_levels[high_index]])
        return chip_range
    
    def calculate_winner(self, current_prices):
        total_chips = np.sum(self.chip_distribution, axis=1)
        winners = np.sum(self.chip_distribution[:, :, :] * (self.price_levels[np.newaxis, :, np.newaxis] <= current_prices[:, np.newaxis, :]), axis=1) / total_chips
        return winners

    def calculate_cost(self, percentile):
        costs = np.percentile(self.chip_distribution, percentile, axis=1)
        return costs
    def calculate_distribution(self, winners, bins=np.arange(-90, 100, 10)):
        """
        #亏损区间：<-90% 和 [-90%, -80%], ..., [-10%, 0%]
        # [0%, 10%], [10%, 20%], ..., >90%
        
        bins：np.arange(-90, 100, 10) 创建了从 [-90, -80], ..., [80, 90] 的分组边界。最小边界小于 -90 和最大边界大于 90 的情况单独处理。
less_than_minus_90 和 greater_than_90：分别计算获利小于 -90% 和大于 90% 的筹码数量。
percentage_distribution：计算各个区间（包括小于 -90% 和大于 90% 的边界）的占比。
构建 DataFrame：最终包含 Loss_<-90%, 各个区间 [0%, 10%], ..., 以及 Profit_>90% 的占比。

        Parameters
        ----------
        winners : TYPE
            DESCRIPTION.
        bins : TYPE, optional
            DESCRIPTION. The default is np.arange(-90, 100, 10).

        Returns
        -------
        Stock_ID  Time_Step  Loss_<-90%  Range_-90_-80  Range_-80_-70  ...  Range_80_90  Profit_>90%
0         1          1         0.0           0.0           0.1  ...         0.0          0.0
1         1          2         0.0           0.0           0.1  ...         0.0          0.0
2         1          3         0.0           0.0           0.1  ...         0.0          0.1
...
.

        """
        num_stocks, num_time_steps = winners.shape
        distributions = []
    
        # 将 Winners 乘以 100 得到百分比收益
        winners_percentage = winners * 100
    
        for stock_id in range(num_stocks):
            for time_step in range(num_time_steps):
                # 分别处理 -∞ 到 -90 和 90 到 ∞ 的区间
                less_than_minus_90 = np.sum(winners_percentage[stock_id, time_step] < -90)
                greater_than_90 = np.sum(winners_percentage[stock_id, time_step] > 90)
                
                # 处理其余的区间 [-90, -80], [-80, -70], ..., [80, 90]
                counts, _ = np.histogram(winners_percentage[stock_id, time_step], bins=bins, density=False)
                total_count = less_than_minus_90 + greater_than_90 + np.sum(counts)
                
                # 计算各个区间的百分比
                percentage_distribution = np.hstack(([less_than_minus_90], counts, [greater_than_90])) / total_count if total_count != 0 else np.zeros(len(counts) + 2)
                
                # 构建字典
                dist_data = {
                    'Stock_ID': stock_id + 1,
                    'Time_Step': time_step + 1
                }
                
                # 添加每个区间的占比
                dist_data['Loss_<-90%'] = percentage_distribution[0]
                for i in range(1, len(bins)):
                    dist_data[f'Range_{bins[i-1]}_{bins[i]}'] = percentage_distribution[i]
                dist_data['Profit_>90%'] = percentage_distribution[-1]
    
                distributions.append(dist_data)
        
        df = pd.DataFrame(distributions)
        return df

# 读取和处理CSV文件
def process_stock_data(file_path):
    df = pd.read_csv(file_path)
    return df[['high', 'low', 'avg', 'volume', 'TurnoverRate']]

# 主函数
def main(data):
    num_stocks = 5
    price_range = 100
    time_steps = 10

    # 初始化优化器
    optimizer = ChipDistributionVectorized(num_stocks, price_range, time_steps)

    # 假设我们已经从文件中读取了DataFrame
    stock_data = pd.DataFrame(data)  # 假设 'data' 是之前定义的股票数据

    # 更新分布
    optimizer.update_distribution(stock_data)

    # 假设当前价格
    current_prices = np.random.uniform(50, 150, (num_stocks, time_steps))

    # 计算赢家和成本
    winners = optimizer.calculate_winner(current_prices)
    costs = optimizer.calculate_cost(90)
    
    num_stocks, num_time_steps = winners.shape
    data = []

    for stock_id in range(num_stocks):
        for time_step in range(num_time_steps):
            data.append({
               'Stock_ID': stock_id + 1,
               'Time_Step': time_step + 1,
               'Winner': winners[stock_id, time_step],
               'Cost': costs[stock_id, time_step]
            })
    
    df = pd.DataFrame(data)
#    print("Winners:\n", winners)
#    print("Costs:\n", costs)
    print(df)
    
    # 假设 winners 是一个 5×10 的 NumPy 数组（5 只股票在 10 个时间步的获利情况）
# =============================================================================
#     winners1 = np.array([[-0.00062434, -0.00127082, -0.069222, -0.0586944, -0.05172621, -0.04241737, -0.03021219, -0.04346632, -0.05082945, -0.04639309],
#                     [0.00454756, 0.00405304, 0.00474644, 0.00181998, 0.00231379, 0.00386999, 0.00237723, 0.00144086, 0.00168769, 0.00241919],
#                     [0.00352299, 0.00493054, 0.01477529, 0.0157403, -0.38700904, -0.37270468, -0.29253152, -0.27059867, -0.23510849, -0.22381806],
#                     [0.01802723, 0.0486951, 0.01666872, 0.19445827, 0.15761652, 0.14464712, 0.13383158, 0.11729723, 0.09830157, 0.07967283],
#                     [0.05840521, 0.03831404, 0.03036593, 0.0255213, 0.0189792, 0.01981535, 0.0134436, 0.0153406, 0.01349644, 0.01008482]])
# 
# =============================================================================
    df_distribution = optimizer.calculate_distribution(winners)
    print(df_distribution)
    
if __name__ == "__main__":
    # 模拟股票数据，每只股票有若干时间步的high、low、avg、volume、TurnoverRate
    data = {
        'stock_id': np.repeat(np.arange(5), 10),  # 5只股票，每只股票10个时间步
        'time_step': np.tile(np.arange(10), 5),  # 10个时间步
        'high': np.random.uniform(100, 200, 50),
        'low': np.random.uniform(50, 100, 50),
        'avg': np.random.uniform(75, 150, 50),
        'volume': np.random.uniform(1000, 5000, 50),
        'TurnoverRate': np.random.uniform(0.01, 0.1, 50)
    }
    #df = pd.DataFrame(data)
    main(data)