# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:07:57 2024

@author: awei
(vap4)
Numba JIT编译，计算速度更快
减少循环，增加向量化计算
内存使用更高效
"""
import os
import numpy as np
import pandas as pd
import numba
from typing import List, Dict, Union
import os
import argparse

import pandas as pd
import numpy as np

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

@numba.njit
def vectorized_chip_distribution(
    chip_dist: np.ndarray, 
    high: float, 
    low: float, 
    avg_price: float, 
    vol: float, 
    turnover: float, 
    price_levels: np.ndarray, 
    price_step: float
) -> np.ndarray:
    """
    Vectorized chip distribution calculation with Numba JIT compilation
    """
    # Initialize new distribution array
    new_dist = np.zeros_like(chip_dist)
    
    # Triangular distribution calculation with division-by-zero check
    if high != low:
        h = 2 / (high - low)
        new_dist = np.where(
            price_levels < avg_price,
            h / (avg_price - low) * (price_levels - low),
            h / (high - avg_price) * (high - price_levels)
        )
    
    # Normalize new_dist and apply volume if the sum of new_dist is non-zero
    dist_sum = np.sum(new_dist)
    if dist_sum > 0 and vol > 0 and turnover > 0:
        new_dist = new_dist / dist_sum * vol * turnover
    else:
        new_dist = np.zeros_like(new_dist)  # No distribution if normalization conditions fail

    # Update chip distribution
    chip_dist = chip_dist * (1 - turnover) + new_dist
    return chip_dist


class ChipDistributionAnalyzer:
    def __init__(
        self, 
        price_range: int = 100, 
        min_price: float = 0, 
        max_price: float = 1000
    ):
        """
        Initialize chip distribution analyzer
        
        Args:
            price_range (int): Number of price levels
            min_price (float): Minimum price
            max_price (float): Maximum price
        """
        self.price_range = price_range
        self.min_price = min_price
        self.max_price = max_price
        self.price_step = (max_price - min_price) / price_range
        self.price_levels = np.linspace(min_price, max_price, price_range)
    
    def calculate_chip_distribution(
        self, 
        stock_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate chip distribution for multiple stocks
        
        Args:
            stock_data (pd.DataFrame): Stock data with columns 
            ['full_code', 'date', 'high', 'low', 'avg_price', 'volume', 'turnover']
        
        Returns:
            pd.DataFrame: Chip distribution results
        """
        # Group by stock and sort by date
        grouped_data = stock_data.sort_values(['full_code', 'date'])
        
        # Prepare result storage
        results = []
        
        for stock_code, group in grouped_data.groupby('full_code'):
            # Initialize chip distribution
            chip_dist = np.zeros((self.price_range, len(group)))
            
            # Calculate chip distribution for each time step
            for t, row in group.iterrows():
                if t == group.index[0]:
                    # First time step
                    chip_dist[:, 0] = vectorized_chip_distribution(
                        np.zeros(self.price_range),
                        row['high'], row['low'], row['avg_price'], 
                        row['volume'], row['turnover'], 
                        self.price_levels, self.price_step
                    )
                else:
                    # Subsequent time steps
                    chip_dist[:, t-group.index[0]] = vectorized_chip_distribution(
                        chip_dist[:, t-group.index[0]-1],
                        row['high'], row['low'], row['avg_price'], 
                        row['volume'], row['turnover'], 
                        self.price_levels, self.price_step
                    )
            
            # Process results
            for t in range(len(group)):
                results.append({
                    'full_code': stock_code,
                    'date': group.iloc[t]['date'],
                    'chip_dist': chip_dist[:, t]
                })
        
        return pd.DataFrame(results)
    
    def analyze_chip_distribution(
        self, 
        chip_dist_df: pd.DataFrame, 
        current_prices: Union[pd.Series, np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze chip distribution
        
        Args:
            chip_dist_df (pd.DataFrame): Chip distribution DataFrame
            current_prices (Series/ndarray, optional): Current prices for each stock
        
        Returns:
            pd.DataFrame: Analysis results
        """
        results = []
        
        for _, row in chip_dist_df.iterrows():
            dist = row['chip_dist']
            
            # Calculate statistics
            result = {
                'full_code': row['full_code'],
                'date': row['date'],
                'total_chips': np.sum(dist),
                'median_price': np.percentile(self.price_levels, 50),
                'cost_90_percentile': np.percentile(self.price_levels, 90),
                'cost_10_percentile': np.percentile(self.price_levels, 10)
            }
            
            # If current prices provided, calculate winners
            if current_prices is not None:
                total_chips = np.sum(dist)
                winner = np.sum(dist * (self.price_levels <= current_prices[row['full_code']])) / total_chips
                result['winner'] = winner
            
            results.append(result)
        
        return pd.DataFrame(results)


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
    #stock_daily_df['turnover'] = stock_daily_df['turnover']/100
    analyzer = ChipDistributionAnalyzer()
    
    # Calculate chip distribution
    chip_dist_df = analyzer.calculate_chip_distribution(stock_daily_df)
    
    # Analyze chip distribution
    current_prices = {code: np.random.uniform(30, 90) for code in stock_daily_df['full_code'].unique()}
    results = analyzer.analyze_chip_distribution(
        chip_dist_df, 
        current_prices=current_prices
    )
    
    print(results)
    
    #['full_code', 'date', 'total_chips', 'median_price',
    #       'cost_90_percentile', 'cost_10_percentile', 'winner']