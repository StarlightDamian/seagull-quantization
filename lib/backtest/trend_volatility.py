# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:57:12 2024

@author: awei
trend_volatility
趋势trend
震荡volatility 

a) ADX (Average Directional Index)：

- ADX值范围从0到100
- 通常认为ADX > 25表示强趋势，ADX < 20表示弱趋势或震荡市场
- 方向可以通过比较+DI和-DI来确定

b) Choppiness Index (CI)：

- 值范围从0到100
- 接近100表示高度震荡，接近0表示强趋势
- 不提供方向信息

c) Volatility Ratio：

- 比较短期和长期波动率
- 值大于1表示趋势市场，小于1表示震荡市场
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

def calculate_choppiness_index(high, low, close, period=14):
    tr = vbt.indicators.ATR.run(high, low, close, window=1).atr.to_numpy()
    atr_sum = vbt.generic.nb.rolling_sum_nb(tr, period)
    highest_high = vbt.generic.nb.rolling_max_nb(high.to_numpy(), period)
    lowest_low = vbt.generic.nb.rolling_min_nb(low.to_numpy(), period)
    
    ci = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return pd.Series(ci, index=close.index)

def calculate_adx(high, low, close, period=14):
    adx = vbt.indicators.ADX.run(high, low, close, window=period).adx.to_numpy()
    return pd.Series(adx, index=close.index)

def market_state_indicator(high, low, close, adx_period=14, ci_period=14):
    adx = calculate_adx(high, low, close, adx_period)
    ci = calculate_choppiness_index(high, low, close, ci_period)
    
    # Normalize ADX and CI to a 0-1 scale
    adx_norm = adx / 100
    ci_norm = ci / 100
    
    # Create a composite indicator
    # Higher values indicate stronger trends, lower values indicate choppiness
    composite = (adx_norm + (1 - ci_norm)) / 2
    
    # Determine direction
    direction = np.where(close.pct_change() > 0, 1, -1)
    
    return pd.DataFrame({
        'ADX': adx,
        'CI': ci,
        'Composite': composite,
        'Direction': direction
    }, index=close.index)

# Example usage
def analyze_market_state(symbol, start_date, end_date):
    # Fetch data
    data = vbt.YFData.download(symbol, start=start_date, end=end_date)
    
    # Calculate market state indicator
    msi = market_state_indicator(data.high, data.low, data.close)
    
    # Define market states
    msi['State'] = np.select(
        [
            (msi['Composite'] > 0.7) & (msi['Direction'] == 1),
            (msi['Composite'] > 0.7) & (msi['Direction'] == -1),
            (msi['Composite'] < 0.3),
        ],
        ['Strong Uptrend', 'Strong Downtrend', 'Choppy'],
        default='Neutral'
    )
    
    # Plot results
    fig = vbt.plotting.Figure()
    fig.add_scatter(x=msi.index, y=data.close, name='Close Price')
    fig.add_scatter(x=msi.index, y=msi['Composite'], name='Market State Indicator', yaxis='y2')
    fig.update_layout(title=f'Market State Analysis for {symbol}',
                      yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    fig.show()
    
    # Print summary
    print(msi['State'].value_counts(normalize=True))

# Run the analysis
analyze_market_state('AAPL', '2020-01-01', '2023-12-31')