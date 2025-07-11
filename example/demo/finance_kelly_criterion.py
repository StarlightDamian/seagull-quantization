# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:21:41 2024

@author: awei
凯利公式(finance_Kelly_criterion)
"""
import pandas as pd

def kelly_criterion(df):
    """
    计算凯利公式推荐的投注比例，并返回包含推荐投注比例的 DataFrame。
    
    参数:
    - df: 包含赔率和胜率的 DataFrame。需要有 'odds' 和 'win_prob' 列。
    
    返回:
    - result_df: 包含原始数据和计算出的推荐投注比例 'kelly_fraction' 的 DataFrame。
    """
    # 计算败率 q
    df['lose_prob'] = 1 - df['win_prob']
    
    # 使用凯利公式计算最佳投注比例
    df['kelly_fraction'] = (df['odds'] * df['win_prob'] - df['lose_prob']) / df['odds']
    
    # 将投注比例限制在0到1之间，避免过度风险
    df['kelly_fraction'] = df['kelly_fraction'].clip(lower=0, upper=1)
    
    return df

# 示例 DataFrame
data = {
    'odds': [2.0, 3.5, 5.0, 1.8],     # 赔率（期望获利倍数）
    'win_prob': [0.6, 0.4, 0.3, 0.7]  # 胜率（获胜的概率）
}

# 创建 DataFrame 并计算凯利比例
df = pd.DataFrame(data)
result_df = kelly_criterion(df)
print(result_df)
