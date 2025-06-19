# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:04:22 2025

@author: awei
icir
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from seagull.settings import PATH

class FactorAnalyzer:
    def __init__(self):
        self.results = {}
    
    def winsorize(self, data, n=3):
        """
        极值处理
        """
        mean = data.mean()
        std = data.std()
        upper = mean + n * std
        lower = mean - n * std
        return np.clip(data, lower, upper)
    
    def neutralize(self, factor_data, market_cap, industry_dummies):
        """
        市值、行业中性化
        """
        # 准备回归变量
        X = pd.concat([np.log(market_cap), industry_dummies], axis=1)
        X = sm.add_constant(X)
        
        # 对每个时间点进行横截面回归
        residuals = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
        
        for date in factor_data.columns:
            y = factor_data.loc[date]
            mask = ~(y.isna() | X.loc[date].isna().any(axis=1))
            if mask.sum() > 0:
                model = sm.OLS(y[mask], X.loc[date][mask], missing='drop')
                residuals.loc[date] = model.fit().resid
        
        return residuals
    
    def standardize(self, data):
        """
        标准化处理
        按特征列处理
        """
        return (data - data.mean()) / data.std()
    
    def calculate_ic(self, factor_data, forward_returns, method='spearman'):
        """
        计算IC值
        """
        ic_series = pd.Series(index=factor_data.columns)
        
        for date in factor_data.columns:
            if method == 'spearman':
                ic = stats.spearmanr(factor_data[date], forward_returns[date])[0]
            else:  # pearson
                ic = stats.pearsonr(factor_data[date], forward_returns[date])[0]
            ic_series[date] = ic
            
        return ic_series
    
    def calculate_icir(self, ic_series):
        """
        计算ICIR
        """
        return ic_series.mean() / ic_series.std()
    
    def analyze_factor(self, factor_data, forward_returns, market_cap=None, 
                      industry_dummies=None, winsorize=True, neutralize=True):
        """
        完整的因子分析流程
        """
        processed_data = factor_data.copy()
        
        # 1. 去极值
        if winsorize:
            processed_data = processed_data.apply(self.winsorize)
        
        # 2. 中性化
        if neutralize and market_cap is not None and industry_dummies is not None:
            processed_data = self.neutralize(processed_data, market_cap, industry_dummies)
            
        # 3. 标准化
        processed_data = processed_data.apply(self.standardize)
        
        # 4. 计算IC和ICIR
        ic_series = self.calculate_ic(processed_data, forward_returns)
        icir = self.calculate_icir(ic_series)
        
        # 存储结果
        self.results = {
            'IC_series': ic_series,
            'IC_mean': ic_series.mean(),
            'IC_std': ic_series.std(),
            'ICIR': icir,
            'IC_positive_rate': (ic_series > 0).mean()
        }
        
        return self.results

# 使用示例
if __name__ == "__main__":
    # 假设数据
    raw_df = pd.read_feather(f'{PATH}/_file/das_wide_incr_train.feather')
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    stocks = [f'stock_{i}' for i in range(100)]
    
    # 创建样本数据
    factor_data = pd.DataFrame(np.random.randn(len(stocks), len(dates)), 
                             index=stocks, columns=dates)
    forward_returns = pd.DataFrame(np.random.randn(len(stocks), len(dates)), 
                                 index=stocks, columns=dates)
    
    # 初始化分析器
    analyzer = FactorAnalyzer()
    
    # 运行分析
    results = analyzer.analyze_factor(factor_data, forward_returns)
    print("Analysis Results:")
    for key, value in results.items():
        print(f"{key}:", value)