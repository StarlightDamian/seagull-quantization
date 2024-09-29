# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:04:08 2024

@author: awei
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

# Assume we have a price DataFrame 'data' and a portfolio object 'portfolio'
data = vbt.YFData.download('AAPL', start='2020-01-01', end='2021-01-01').get('Close')

import pandas as pd
import numpy as np
import vectorbt as vbt

def get_accurate_dates(data):
    first_valid = data.apply(lambda col: col.first_valid_index())
    last_valid = data.apply(lambda col: col.last_valid_index())
    return pd.DataFrame({'start': first_valid, 'end': last_valid})

def calculate_accurate_stats(portfolio, data):
    dates = get_accurate_dates(data)
    stats = portfolio.stats(agg_func=None)
    
    for col in stats.columns:
        start = dates.loc[col, 'start']
        end = dates.loc[col, 'end']
        
        # Update start and end
        stats.loc['start', col] = start
        stats.loc['end', col] = end
        
        # Recalculate total return
        returns = portfolio.returns().loc[start:end, col]
        total_return = (1 + returns).prod() - 1
        stats.loc['total_return', col] = total_return
        
        # Recalculate annualized return
        years = (end - start).days / 365.25
        stats.loc['annualized_return', col] = (1 + total_return) ** (1 / years) - 1
        
        # Recalculate Sharpe ratio
        risk_free_rate = 0  # Adjust as needed
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Assuming daily data
        stats.loc['sharpe_ratio', col] = sharpe_ratio
    
    return stats

# Create portfolio
portfolio = vbt.Portfolio.from_holding(data)

# Calculate accurate stats
accurate_stats = calculate_accurate_stats(portfolio, data)

print("Accurate Stats:")
print(accurate_stats)

# Visualization
def plot_portfolio_with_accurate_dates(portfolio, data):
    dates = get_accurate_dates(data)
    fig = portfolio.plot()
    
    for col in data.columns:
        start = dates.loc[col, 'start']
        end = dates.loc[col, 'end']
        fig.add_vrect(
            x0=start, 
            x1=end, 
            fillcolor=f"rgba({hash(col) % 256}, {(hash(col) >> 8) % 256}, {(hash(col) >> 16) % 256}, 0.1)", 
            layer="below", 
            line_width=0
        )
    
    fig.show()

#plot_portfolio_with_accurate_dates(portfolio, data)).show()