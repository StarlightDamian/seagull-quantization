# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:20:45 2024

@author: awei
plot_bokeh
"""

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import output_notebook
import pandas as pd
import numpy as np

# 生成示例数据
def generate_sample_data(n=100):
    dates = pd.date_range(start='2024-01-01', periods=n)
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 2, n),
        'high': None,
        'low': None,
        'close': None
    })
    
    # 生成高低收盘价
    for i in range(len(data)):
        data.loc[i, 'close'] = data.loc[i, 'open'] + np.random.normal(0, 2)
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close']) + abs(np.random.normal(0, 1))
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close']) - abs(np.random.normal(0, 1))
    
    return data

def create_candlestick_chart(data):
    # 创建图形
    p = figure(width=1000, height=600, x_axis_type="datetime", title="蜡烛图交易区域分析")
    
    # 设置图形样式
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = '日期'
    p.yaxis.axis_label = '价格'
    
    # 计算上涨和下跌
    inc = data.close > data.open
    dec = data.close < data.open
    
    # 绘制蜡烛图
    # 上涨蜡烛
    p.segment(data.date[inc], data.high[inc], data.date[inc], data.low[inc], color='red')
    p.vbar(data.date[inc], 0.5, data.open[inc], data.close[inc], fill_color='red', line_color='red')
    
    # 下跌蜡烛
    p.segment(data.date[dec], data.high[dec], data.date[dec], data.low[dec], color='green')
    p.vbar(data.date[dec], 0.5, data.open[dec], data.close[dec], fill_color='green', line_color='green')
    
    # 添加移动平均线
    ma20 = data.close.rolling(window=20).mean()
    ma60 = data.close.rolling(window=60).mean()
    
    p.line(data.date, ma20, color='blue', legend_label='MA20', line_width=2)
    p.line(data.date, ma60, color='orange', legend_label='MA60', line_width=2)
    
    # 判断交易区域
    # 当短期均线上穿长期均线时，标记为买入区域
    buy_signal = (ma20.shift(1) < ma60.shift(1)) & (ma20 > ma60)
    sell_signal = (ma20.shift(1) > ma60.shift(1)) & (ma20 < ma60)
    
    # 标记买入区域
    buy_dates = data.date[buy_signal]
    buy_prices = data.low[buy_signal]
    p.circle(buy_dates, buy_prices, size=10, color='red', alpha=0.5, legend_label='买入信号')
    
    # 标记卖出区域
    sell_dates = data.date[sell_signal]
    sell_prices = data.high[sell_signal]
    p.circle(sell_dates, sell_prices, size=10, color='green', alpha=0.5, legend_label='卖出信号')
    
    # 设置图例
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p

# 生成示例数据并创建图表
data = generate_sample_data(100)
p = create_candlestick_chart(data)

# 显示图表
show(p)