# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:45 2024

@author: awei
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import vectorbt as vbt

# df in：'open', 'high', 'low', 'close', 'volume'

def calculate_features(df):
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd - signal
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['MA20'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['MA20'] - 2 * df['close'].rolling(window=20).std()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA20']
    
    # KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    
    return df

def prepare_data(df):
    df = calculate_features(df)
    
    # 创建目标变量：未来N天的收益率
    n_days = 5  # 可以根据需要调整
    df['target'] = df['close'].pct_change(n_days).shift(-n_days)
    
    # 选择特征
    features = ['MACD', 'RSI', 'BB_width', 'K', 'D', 'J', 'MA5', 'MA10']
    X = df[features]
    y = df['target']
    
    # 去除NaN值
    X = X.dropna()
    y = y.loc[X.index]
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def backtest(df, model, scaler):
    X, _ = prepare_data(df)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    df['prediction'] = pd.Series(predictions, index=X.index)
    
    # 创建交易信号
    df['signal'] = np.where(df['prediction'] > 0.02, 1, 0)  # 买入信号阈值可以调整
    
    # 使用vectorbt进行回测
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=df['signal'] == 1,
        exits=df['signal'] == 0,
        init_cash=100000,
        fees=0.001
    )
    
    return portfolio

# 主程序
df = pd.read_csv('your_stock_data.csv')  # 替换为您的数据文件
X, y = prepare_data(df)
model, scaler = train_model(X, y)
portfolio = backtest(df, model, scaler)

# 打印回测结果
print(portfolio.total_return())
print(portfolio.sharpe_ratio())
print(portfolio.max_drawdown())