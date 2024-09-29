# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:07:09 2023

@author: awei
特征工程主程序(feature_engineering_main)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch

from __init__ import path
from base import base_connect_database, base_utils, base_trading_day

pd.options.mode.chained_assignment = None

TARGET_REAL_NAMES = ['rear_low_pct_real', 'rear_high_pct_real', 'rear_diff_pct_real', 'rear_open_pct_real', 'rear_close_pct_real']  # , 'rear_rise_pct_real'


class featureEngineering(base_trading_day.tradingDay):
    def __init__(self, target_names=TARGET_REAL_NAMES):
        """
        初始化函数，用于登录系统和加载行业分类数据
        :param check:是否检修中间层history_day_df
        """
        super().__init__()
        
        # 行业分类数据
        with base_connect_database.engine_conn('postgre') as conn:
            stock_industry = pd.read_sql('stock_industry', con=conn.engine)
        #stock_industry.loc[stock_industry.industry.isnull(), 'industry'] = '其他' # 不能在这步补全，《行业分类数据》不够完整会导致industry为nan
        
        self.code_and_industry_dict = stock_industry.set_index('code')['industry'].to_dict()
        
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.target_names = target_names
        self.price_limit_pct = None
        
    def merge_features_after(self, history_day_df):
        """
        制作待预测值，为后一天的最高价和最低价
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含待预测值的DataFrame
        """
        # debug isST
        last_day = history_day_df.date.unique()[-1]
        last_day_df = history_day_df[history_day_df.date==last_day]
        print('last_day',last_day)
        
        # 待预测的指定交易日的主键、价格
        predict_pd = history_day_df[['date_before', 'code', 'open', 'low', 'high', 'close','price_limit']]
        try:
            predict_pd = predict_pd[~(predict_pd.date_before.isnull())]
            predict_pd['primary_key'] = (predict_pd['date_before']+predict_pd['code']).apply(base_utils.md5_str)
        except:
            print('没有获取交易日最新数据')
            
        predict_pd = predict_pd.rename(columns={'open': 'rear_open',
                                                'low': 'rear_low',
                                                'high': 'rear_high',
                                                'close': 'rear_close',
                                                'price_limit': 'rear_price_limit',
                                                })
        predict_pd = predict_pd[['primary_key', 'rear_high', 'rear_low', 'rear_open', 'rear_close','rear_price_limit']]
        
        # 关联对应后置最低最高价格
        history_day_df = pd.merge(history_day_df, predict_pd, on='primary_key')
        
        ## 制作待预测值
        # 明日最高/低值相对于今日收盘价的涨跌幅真实值
        history_day_df['rear_low_pct_real'] = ((history_day_df['rear_low'] - history_day_df['close']) / history_day_df['close']) * 100
        history_day_df['rear_high_pct_real'] = ((history_day_df['rear_high'] - history_day_df['close']) / history_day_df['close']) * 100
        history_day_df['rear_diff_pct_real'] = history_day_df.rear_high_pct_real - history_day_df.rear_low_pct_real
        history_day_df['rear_open_pct_real'] = ((history_day_df['rear_open'] - history_day_df['close']) / history_day_df['close']) * 100
        history_day_df['rear_close_pct_real'] = ((history_day_df['rear_close'] - history_day_df['close']) / history_day_df['close']) * 100
        #history_day_df['rear_rise_pct_real'] = ((history_day_df['rear_high'] - history_day_df['low']) / history_day_df['low']) * 100
        
        last_day_df[['rear_low_pct_real', 'rear_high_pct_real', 'rear_diff_pct_real', 'rear_open_pct_real', 'rear_close_pct_real']] = 0.0  # 最后一天：真实数据预测，y设置为0.0., 'rear_rise_pct_real'
        
        last_day_df[['rear_price_limit']] = 0
        history_day_df = pd.concat([history_day_df, last_day_df],axis=0)
        return history_day_df
    
    
    def build_features_after(self, history_day_df):
        # 特征: 宏观大盘_大盘成交量
        sh000001_map_dict = history_day_df[history_day_df.code=='sh.000001'][['date', 'amount']].set_index('date')['amount'].to_dict()
        history_day_df['macro_amount_sh000001'] = history_day_df['date'].map(sh000001_map_dict)  # 上证综合指数
        
        sz399106_map_dict = history_day_df[history_day_df.code=='sz.399106'][['date', 'amount']].set_index('date')['amount'].to_dict()
        history_day_df['macro_amount_sz399106'] = history_day_df['date'].map(sz399106_map_dict)  # 深证综合指数
        
        history_day_df['macro_amount'] = history_day_df['macro_amount_sh000001'] + history_day_df['macro_amount_sz399106']#两市成交额
        return history_day_df
    
    def merge_features_before(self, history_day_df):
        """
        制作待预测值，为后一天的最高价和最低价
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含待预测值的DataFrame
        """
        # 待预测的指定交易日的主键、价格
        predict_pd = history_day_df[['date_after', 'code', 'macro_amount']]
        
        predict_pd['primary_key'] = (predict_pd['date_after']+predict_pd['code']).apply(base_utils.md5_str)
        predict_pd = predict_pd.rename(columns={'macro_amount': 'pre_macro_amount',
                                                })
        predict_pd = predict_pd[['primary_key', 'pre_macro_amount']]
        # 关联对应后置最低最高价格
        #print('history_day_df',history_day_df.columns,history_day_df.shape,history_day_df.primary_key.tolist())
        #print('predict_pd',predict_pd.columns,predict_pd.shape,predict_pd.primary_key.tolist())
        history_day_df.to_csv(f'{path}/data/history_day_df.csv',index=False)
        predict_pd.to_csv(f'{path}/data/history_day_df.csv',index=False)
        history_day_df = pd.merge(history_day_df, predict_pd, on='primary_key')
        
        #前一天成交额差值
        history_day_df['macro_amount_diff_1'] = history_day_df['macro_amount'] - history_day_df['pre_macro_amount']
        return history_day_df
    
    def build_features_before(self, history_day_df):
        """
        构建数据集，将DataFrame转换为Bunch
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        ## 训练特征
        
        # 特征: 基础_距离上一次开盘天数
        #history_day_df['date_diff'] = (pd.to_datetime(history_day_df.date_before) - pd.to_datetime(history_day_df.date)).dt.days
        
        # 特征：基础_星期
        history_day_df['date_week'] = pd.to_datetime(history_day_df['date'], format='%Y-%m-%d').dt.day_name()
        
        # 特征: 宏观大盘_大盘成交量
        sz399101_map_dict = history_day_df[history_day_df.code=='sz.399101'][['date', 'amount']].set_index('date')['amount'].to_dict()
        history_day_df['macro_amount_sz399101'] = history_day_df['date'].map(sz399101_map_dict) # 中小企业综合指数 
         
        sz399102_map_dict = history_day_df[history_day_df.code=='sz.399102'][['date', 'amount']].set_index('date')['amount'].to_dict()
        history_day_df['macro_amount_sz399102'] = history_day_df['date'].map(sz399102_map_dict) # 创业板综合指数
         
        sh000300_map_dict = history_day_df[history_day_df.code=='sh.000300'][['date', 'amount']].set_index('date')['amount'].to_dict()
        history_day_df['macro_amount_sh000300'] = history_day_df['date'].map(sh000300_map_dict)  # 沪深300指数
         
        # 特征: 宏观大盘_大盘换手率
        sh000001_map_dict = history_day_df[history_day_df.code=='sh.000001'][['date', 'turn']].set_index('date')['turn'].to_dict()
        history_day_df['macro_turn_sh000001'] = history_day_df['date'].map(sh000001_map_dict)  # 上证综合指数
        sz399106_map_dict = history_day_df[history_day_df.code=='sz.399106'][['date', 'turn']].set_index('date')['turn'].to_dict()
        history_day_df['macro_turn_sz399106'] = history_day_df['date'].map(sz399106_map_dict)  # 深证综合指数
         
        # 特征: 中观板块_行业
        history_day_df['industry'] = history_day_df.code.map(self.code_and_industry_dict)
        history_day_df['industry'] = history_day_df['industry'].replace(['', pd.NA], '其他')
         
        # lightgbm不支持str，把str类型转化为ont-hot
        history_day_df = pd.get_dummies(history_day_df, columns=['industry', 'tradestatus',  'date_week']) # 'isST',
        
        feature_names = history_day_df.columns.tolist()
        
        # 删除非训练字段
        columns_to_drop = ['date', 'code', 'code_name', 'adjustflag','isST', 'date_before', 'date_after', 'rear_low', 'rear_high', 'rear_open', 'rear_close', 'pre_macro_amount','board_type','price_limit_pct'] + self.target_names
        feature_names = list(set(feature_names) - set(columns_to_drop))
        feature_names = [x for x in feature_names if '_real' not in x] # rear_
        return history_day_df, feature_names
    
    def build_dataset(self, history_day_df, feature_names, price_limit_pct=None):
        
        num_before = history_day_df.shape[0]
        print(f'处理前数据量: {num_before} |price_limit_pct: {self.price_limit_pct}')
        if self.price_limit_pct!=None:
            history_day_df = history_day_df[(history_day_df.rear_low_pct_real<=self.price_limit_pct)&(history_day_df.rear_low_pct_real>=-self.price_limit_pct)&(history_day_df.rear_high_pct_real<=self.price_limit_pct)&(history_day_df.rear_high_pct_real>=-self.price_limit_pct)]
        
            num_after = history_day_df.shape[0]
            print(f'处理前数据量: {num_before} |处理后数据量: {num_after}')
            
        # 构建数据集
        feature_names = sorted(feature_names) # 输出有序标签
        # print(f'feature_names_engineering:\n {feature_names}')
        
        date_range_dict = {'data': np.array(history_day_df[feature_names].to_records(index=False)),  # 不使用 feature_df.values,使用结构化数组保存每一列的类型
                         'feature_names': feature_names,
                         'target': history_day_df[self.target_names].values,  # 机器学习预测值
                         'target_names': [self.target_names],
                         }
        date_range_bunch = Bunch(**date_range_dict)
        return date_range_bunch
    
    def feature_engineering_pipline(self, history_day_df):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        trading_day_before_dict = self.specified_trading_day_before(pre_date_num=1)
        history_day_df['date_before'] = history_day_df.date.map(trading_day_before_dict)
        #print(history_day_df[['date','date_before']])
        
        trading_day_after_dict = self.specified_trading_day_after(pre_date_num=1)
        history_day_df['date_after'] = history_day_df.date.map(trading_day_after_dict)
        
        #特征: 微观个股_涨跌停标识
        history_day_df['price_limit'] = history_day_df.apply(lambda row: 1 if row['high'] == row['low'] else 0, axis=1)
        
        history_day_df = self.merge_features_after(history_day_df)
        
        history_day_df = self.build_features_after(history_day_df)
        
        history_day_df = self.merge_features_before(history_day_df)
        
        # 构建数据集
        history_day_df, feature_names = self.build_features_before(history_day_df)
        return history_day_df, feature_names
    
    def feature_engineering_dataset_pipline(self, history_day_df, price_limit_pct=None):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param history_day_df: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        # 构建数据集
        self.price_limit_pct = price_limit_pct
        #print('fem',history_day_df)
        history_day_df, feature_names = self.feature_engineering_pipline(history_day_df)
        print('feature_names',feature_names)
        date_range_bunch = self.build_dataset(history_day_df, feature_names)
        return date_range_bunch
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-12-20', help='When to start feature engineering')
    parser.add_argument('--date_end', type=str, default='2023-12-27', help='End time for feature engineering')
    args = parser.parse_args()
    
    print(f'When to start feature engineering: {args.date_start}\nEnd time for feature engineering: {args.date_end}')
    
    # 获取日期段数据
    with base_connect_database.engine_conn('postgre') as conn:
        history_day_df = pd.read_sql(f"SELECT * FROM history_a_stock_day WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    #history_day_df = data_loading.feather_file_merge(args.date_start, args.date_end)
    print(history_day_df)
    
    feature_engineering = featureEngineering()
    
    history_day_feature_df, feature_names = feature_engineering.feature_engineering_pipline(history_day_df)
    
    
# =============================================================================
#     Index(['dateDiff', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn',
#            'tradestatus', 'pctChg', 'isST', 'industry_交通运输', 'industry_休闲服务',
#            'industry_传媒', 'industry_公用事业', 'industry_其他', 'industry_农林牧渔',
#            'industry_化工', 'industry_医药生物', 'industry_商业贸易', 'industry_国防军工',
#            'industry_家用电器', 'industry_建筑材料', 'industry_建筑装饰', 'industry_房地产',
#            'industry_有色金属', 'industry_机械设备', 'industry_汽车', 'industry_电子',
#            'industry_电气设备', 'industry_纺织服装', 'industry_综合', 'industry_计算机',
#            'industry_轻工制造', 'industry_通信', 'industry_采掘', 'industry_钢铁',
#            'industry_银行', 'industry_非银金融', 'industry_食品饮料'],
#           dtype='object')
# =============================================================================
#macro_sz399101_diff
#macro_sh000001_diff

#sh.000002, 上证A股指数,432298594654
#sz.399107, 深证A股指数,356732648847
#sh.000001,上证综合指数,357057491972
#sz.399106, 深证综合指数,432390342915