# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:59:30 2024

@author: awei
analysis_label
"""

import pandas as pd
import numpy as np

# 假设你已经读取了标签表和特征表
tags_df = pd.read_csv('tag.csv', sep='\t', names=['full_code', 'code', 'name', 'industry', 'level', 'update_time'])
asset_df = pd.read_csv('features.csv', names=['primary_key', 'date', 'datetime', 'type', 'full_code', 'code', 'market', 'name'])

# 数据预处理
def process_tags_and_features(tags_df, asset_df):
    # 清理industry列中的空值
    tags_df['industry'] = tags_df['industry'].fillna('未知')
    
    # 生成one-hot编码
    # 方法1：按行业one-hot
    industry_onehot = pd.get_dummies(tags_df['industry'], prefix='industry')
    
    # 方法2：多标签one-hot（如果一个股票可能有多个标签）
    multi_label_onehot = tags_df.groupby('full_code').apply(
        lambda x: pd.Series(1, index=set(x['industry']))
    ).reindex(columns=set(tags_df['industry']), fill_value=0)
    multi_label_onehot.columns = [f'industry_{col}' for col in multi_label_onehot.columns]
    
    # 合并标签信息到特征表
    # 使用merge并保留特征表的所有记录
    merged_df = asset_df.merge(
        tags_df[['full_code', 'industry', 'name']].drop_duplicates(), 
        on='full_code', 
        how='left'
    )
    
    # 添加one-hot编码
    merged_df = merged_df.merge(
        multi_label_onehot.reset_index(), 
        left_on='full_code', 
        right_on='full_code', 
        how='left'
    )
    
    # 填充缺失的one-hot列
    onehot_columns = [col for col in merged_df.columns if col.startswith('industry_')]
    merged_df[onehot_columns] = merged_df[onehot_columns].fillna(0)
    
    return merged_df

# 执行处理
result_df = process_tags_and_features(tags_df, asset_df)

# 额外的数据质量检查
def data_quality_check(result_df):
    # 检查one-hot编码是否正确
    onehot_columns = [col for col in result_df.columns if col.startswith('industry_')]
    print("One-hot列数:", len(onehot_columns))
    
    # 检查每个股票的行业覆盖情况
    industry_coverage = result_df.groupby('full_code')[onehot_columns].max()
    print("\n行业覆盖情况样例:")
    print(industry_coverage.head())
    
    # 检查缺失值
    print("\n缺失值情况:")
    print(result_df[onehot_columns].isna().sum())

# 执行数据质量检查
data_quality_check(result_df)