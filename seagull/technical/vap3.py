# -*- coding: utf-8 -*-
"""
Modified on Tue Oct 22 2024

@author: awei
Adjusted time window to two years and improved code logic and performance.
"""
import os
import argparse

import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log  # , utils_data

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')


class ChipDistributionVectorized:
    def __init__(self, num_stocks, price_range, dates, min_price=0, max_price=1000):
        self.num_stocks = num_stocks
        self.price_range = price_range
        self.dates = dates
        self.min_price = min_price
        self.max_price = max_price
        self.price_step = (max_price - min_price) / price_range
        self.chip_distribution = np.zeros((num_stocks, price_range, dates))
        self.price_levels = np.linspace(min_price, max_price, price_range)

    def update_distribution(self, df):
        # Group by stock and date, then apply vectorized calculations
        for full_code in df['full_code'].unique():
            stock_data = df[df['full_code'] == full_code].copy()
            for t in range(self.dates):
                high, low, avg_price, vol, turnover = stock_data.iloc[t][['high', 'low', 'avg_price', 'volume', 'turnover']].values
                if t > 0:
                    self.chip_distribution[full_code, :, t] = self._update_distribution_single(
                        self.chip_distribution[full_code, :, t-1],
                        high, low, avg_price, vol, turnover, 1.0, self.price_levels, self.price_step
                    )
                else:
                    self.chip_distribution[full_code, :, t] = self._update_distribution_single(
                        np.zeros(self.price_range), high, low, avg_price, vol, turnover, 1.0, self.price_levels, self.price_step
                    )

    def _update_distribution_single(self, chip_dist, high, low, avg_price, vol, turnover, A, price_levels, price_step):
        new_dist = np.zeros_like(chip_dist)
        
        # Calculate the triangular distribution
        h = 2 / (high - low)
        new_dist = np.where(price_levels < avg_price,
                            h / (avg_price - low) * (price_levels - low),
                            h / (high - avg_price) * (high - price_levels))
        
        # Normalize and apply volume
        new_dist = new_dist / np.sum(new_dist) * vol * turnover * A
        
        # Update the chip distribution
        chip_dist = chip_dist * (1 - turnover * A) + new_dist
        return chip_dist
    
    def calculate_chip_range(self, percentile=90):
        """
        Calculate the price range for the given percentile of chips.
        """
        chip_range = []
        for full_code in range(self.num_stocks):
            for t in range(self.dates):
                cumulative_distribution = np.cumsum(self.chip_distribution[full_code, :, t])
                low_index = np.searchsorted(cumulative_distribution, (100 - percentile) / 100)
                high_index = np.searchsorted(cumulative_distribution, percentile / 100)
                chip_range.append([self.price_levels[low_index], self.price_levels[high_index]])
        return chip_range
    
    def calculate_winner(self, current_prices):
        total_chips = np.sum(self.chip_distribution, axis=1)
        winners = np.sum(self.chip_distribution * (self.price_levels[np.newaxis, :, np.newaxis] <= current_prices[:, np.newaxis, :]), axis=1) / total_chips
        return winners

    def calculate_cost(self, percentile):
        costs = np.percentile(self.chip_distribution, percentile, axis=1)
        return costs
    
    def calculate_distribution(self, winners, bins=np.arange(-90, 100, 10)):
        """
        Calculate percentage distribution in different profit/loss ranges.
        """
        num_stocks, num_dates = winners.shape
        distributions = []
    
        # Convert Winners to percentage
        winners_percentage = winners * 100
    
        for full_code in range(num_stocks):
            for date in range(num_dates):
                # Handle less than -90% and greater than 90%
                less_than_minus_90 = np.sum(winners_percentage[full_code, date] < -90)
                greater_than_90 = np.sum(winners_percentage[full_code, date] > 90)
                
                # Handle ranges [-90, -80], [-80, -70], ..., [80, 90]
                counts, _ = np.histogram(winners_percentage[full_code, date], bins=bins, density=False)
                total_count = less_than_minus_90 + greater_than_90 + np.sum(counts)
                
                # Calculate percentage distribution
                percentage_distribution = np.hstack(([less_than_minus_90], counts, [greater_than_90])) / total_count if total_count != 0 else np.zeros(len(counts) + 2)
                
                # Build dictionary for each stock and time step
                dist_data = {
                    'full_code': full_code + 1,
                    'date': date + 1
                }
                
                # Add percentage for each range
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
    return df[['high', 'low', 'avg_price', 'volume', 'turnover']]

    
if __name__ == '__main__':
    # date in 1990-12-19, 2024-08-14
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
            AND full_code in ('510300.sh', '601919.sh')
                                     """, con=conn.engine)
    stock_daily_df.drop_duplicates('primary_key',keep='first', inplace=True)
    # avg_price = value_traded / volume
    
    num_stocks = 5
    price_range = 100
    dates = 504  # Two years of daily trading days (252 days per year)

    # Initialize the optimizer
    optimizer = ChipDistributionVectorized(num_stocks, price_range, dates)

    # Update distribution
    optimizer.update_distribution(stock_daily_df)

    # Assume current prices for each stock at each time step
    current_prices = np.random.uniform(50, 150, (num_stocks, dates))

    # Calculate winners and costs
    winners = optimizer.calculate_winner(current_prices)
    costs = optimizer.calculate_cost(90)
    
    num_stocks, num_dates = winners.shape
    data = []

    for full_code in range(num_stocks):
        for date in range(num_dates):
            data.append({
               'full_code': full_code + 1,
               'date': date + 1,
               'winner': winners[full_code, date],
               'cost': costs[full_code, date]
            })
    
    df = pd.DataFrame(data)
    print(df)
    
    # Calculate distribution
    df_distribution = optimizer.calculate_distribution(winners)
    print(df_distribution)
    
# =============================================================================
#     data = {
#         'full_code': np.repeat(np.arange(5), 504),  # 5 stocks, each with 504 time steps (2 years)
#         'date': np.tile(np.arange(504), 5),
#         'high': np.random.uniform(100, 200, 2520),
#         'low': np.random.uniform(50, 100, 2520),
#         'avg_price': np.random.uniform(75, 150, 2520),
#         'volume': np.random.uniform(1000, 5000, 2520),
#         'turnover': np.random.uniform(0.01, 0.1, 2520)
#     }
# =============================================================================
    