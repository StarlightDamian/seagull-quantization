# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:39:53 2025

@author: awei
rolling_cv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class RollingCV(BaseCrossValidator):
    def __init__(self, n_splits=5, train_days=60, gap_days=2, val_rate=0.2):
        # 44+44+8+2=98<100
        self.n_splits = n_splits
        self.train_days = train_days
        self.gap_days = gap_days
        self.val_rate = val_rate
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        total_len = len(X)
        test_size = int(total_len * self.val_rate)
        
        for i in range(self.n_splits):
            # 设置训练集结束和测试集开始的位置
            train_end = self.train_days + (i * (self.train_days + self.gap_days))
            test_start = train_end + self.gap_days
            test_end = test_start + test_size
            
            if test_end > total_len:
                break

            # 获取训练集和测试集的索引
            train_idx = np.arange(train_end - self.train_days, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


# 滚动交叉验证函数
def rolling_window_split(date_series, train_days=22, gap_days=2, val_rate=0.2, n_splits=5):
    data = pd.DataFrame({
        'date': date_series,
    })
    data.set_index('date', inplace=True)
    
    results = []
    total_data_len = len(data)
    
    # 验证集的长度
    val_len = int(train_days * val_rate)

    # 遍历每次滚动的训练和验证集划分
    for i in range(n_splits):
        # 计算训练集的结束日期和验证集的开始日期
        train_end_index = (i + 1) * train_days
        val_start_index = train_end_index + gap_days
        val_end_index = val_start_index + val_len
        
        if val_end_index > total_data_len:
            break  # 如果超出数据范围，停止

        # 获取训练集和验证集
        train_data = data.iloc[train_end_index - train_days: train_end_index]
        val_data = data.iloc[val_start_index: val_end_index]

        # 将训练集和验证集的索引及数据保存到结果中
        results.append({
            'train_start': train_data.index[0],#.strftime('%Y-%m-%d'),
            'train_end': train_data.index[-1],#.strftime('%Y-%m-%d'),
            'val_start': val_data.index[0],#.strftime('%Y-%m-%d'),
            'val_end': val_data.index[-1],#.strftime('%Y-%m-%d'),
            'len_train': train_data.shape[0],
            'len_gap': gap_days,
            'len_val': val_data.shape[0],
        })
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
# =============================================================================
#     # 示例：生成一些时间序列数据
#     np.random.seed(42)
#     date_series = pd.date_range('2024-01-01', periods=100, freq='D')
# 
#     
#     # 设置滚动交叉验证的参数
#     train_days = 30  # 训练集的长度
#     gap_days = 2  # 训练集和验证集之间的间隔天数
#     val_rate = 0.2  # 验证集的大小，通常设置为训练集大小的比例,取下限
#     n_splits = 2  # 交叉验证的次数
#     
#     # 调用滚动交叉验证函数
#     results_df = rolling_window_split(date_series, train_days=train_days, gap_days=gap_days, val_rate=val_rate, n_splits=n_splits)
#     print(results_df)
# =============================================================================
    
    rolling_cv = RollingCV(n_splits=3, train_days=180, gap_days=2, val_rate=0.2)
