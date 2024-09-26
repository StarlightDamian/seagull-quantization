# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:33:32 2024

@author: awei
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt

# 假设 portfolio_strategy 是一个包含策略结果的 DataFrame
# 这里我们创建一个示例 DataFrame
dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
portfolio_strategy = pd.DataFrame({
    'Returns': np.random.randn(len(dates)) * 0.01,
    'Cumulative Returns': np.cumprod(1 + np.random.randn(len(dates)) * 0.01) - 1
}, index=dates)

# 方法1: 使用 Plotly 创建交互式图表并保存为 HTML
def save_plotly_chart(portfolio_strategy, filename):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    
    fig.add_trace(go.Scatter(x=portfolio_strategy.index, y=portfolio_strategy['Returns'], 
                             mode='lines', name='Daily Returns'), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_strategy.index, y=portfolio_strategy['Cumulative Returns'], 
                             mode='lines', name='Cumulative Returns'), row=2, col=1)
    
    fig.update_layout(height=800, title_text="投资组合策略表现")
    fig.write_html(filename)
    print(f"图表已保存到 {filename}")

# 方法2: 使用 Matplotlib 创建静态图表并保存为图片
def save_matplotlib_chart(portfolio_strategy, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(portfolio_strategy.index, portfolio_strategy['Returns'])
    ax1.set_title('Daily Returns')
    ax1.set_ylabel('Returns')
    
    ax2.plot(portfolio_strategy.index, portfolio_strategy['Cumulative Returns'])
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Cumulative Returns')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"图表已保存到 {filename}")

# 使用示例
save_plotly_chart(portfolio_strategy, "./html/portfolio_strategy_plotly.html")
save_matplotlib_chart(portfolio_strategy, "./images/portfolio_strategy_matplotlib.png")

# 如果在支持显示的环境中（如 Jupyter Notebook），可以直接显示
# fig = px.line(portfolio_strategy)
# fig.show()

print("图表已生成。请查看保存的文件。")