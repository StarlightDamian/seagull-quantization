# -*- coding: utf-8 -*-
"""
@Date: 2025/7/9 11:12
@Author: Damian
@Email: zengyuwei1995@163.com
@File: demo_lgb_volatility2.py
@Description: 
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


class TradingLossFunction:
    def __init__(self,
                 trade_threshold=0.02,
                 tail_exponent=1.5,
                 volatility_window=20,
                 asymmetry_ratio=1.2):
        """
        量化交易专用损失函数

        参数:
        trade_threshold: 可交易的最小收益率阈值 (默认2%)
        tail_exponent: 肥尾惩罚指数 (>1)
        volatility_window: 波动率计算窗口
        asymmetry_ratio: 涨跌错误惩罚不对称比例
        """
        self.trade_threshold = trade_threshold
        self.tail_exponent = tail_exponent
        self.volatility_window = volatility_window
        self.asymmetry_ratio = asymmetry_ratio

    def compute_volatility(self, returns):
        """计算自适应波动率"""
        return returns.rolling(self.volatility_window).std().fillna(0.01)

    def weight_function(self, y_true):
        """
        生成样本权重:
        1. 可交易区间样本获得更高权重
        2. 极端事件样本获得指数级权重
        3. 考虑当前波动率水平
        """
        # 基础权重: 可交易区间为1，否则为0.2
        trade_zone = np.where(np.abs(y_true) > self.trade_threshold, 1.0, 0.2)

        # 肥尾惩罚因子: |return|^tail_exponent
        tail_factor = np.abs(y_true) ** self.tail_exponent

        # 波动率调整: 高波动期增加权重
        volatility = self.compute_volatility(y_true)
        vol_factor = np.sqrt(volatility / volatility.mean())

        return trade_zone * tail_factor * vol_factor

    def asymmetric_loss(self, y_true, y_pred):
        """
        非对称损失计算:
        1. 对错过上涨的惩罚 > 对下跌误判的惩罚
        2. 交易区间外使用平滑过渡
        """
        # 预测错误幅度
        error = y_pred - y_true

        # 上涨预测错误惩罚
        up_penalty = np.where(
            (y_true > 0) & (error < 0),
            self.asymmetry_ratio * np.abs(error),
            np.abs(error)
        )

        # 下跌预测错误惩罚
        down_penalty = np.where(
            (y_true < 0) & (error > 0),
            np.abs(error),
            np.abs(error) / self.asymmetry_ratio
        )

        # 组合非对称惩罚
        return np.where(y_true > 0, up_penalty, down_penalty)

    def __call__(self, y_true, y_pred):
        """完整损失计算"""
        # 计算非对称损失基础值
        base_loss = self.asymmetric_loss(y_true, y_pred)

        # 应用样本权重
        weights = self.weight_function(y_true)

        # 避免零除
        weights = np.clip(weights, 1e-6, None)

        return np.mean(base_loss * weights)


import lightgbm as lgb
from scipy.optimize import minimize


class QuantGBMRegressor:
    def __init__(self, loss_params={}, **lgb_params):
        self.loss_fn = TradingLossFunction(**loss_params)
        self.lgb_params = lgb_params

    def _custom_objective(self, preds, train_data):
        """LightGBM自定义目标函数"""
        y_true = train_data.get_label()

        # 计算损失函数值
        loss = self.loss_fn(y_true, preds)

        # 数值梯度计算 (避免解析导数)
        eps = 1e-6
        grad = np.zeros_like(preds)
        for i in range(len(preds)):
            delta = np.zeros_like(preds)
            delta[i] = eps
            grad[i] = (self.loss_fn(y_true, preds + delta) - loss) / eps

        hess = np.ones_like(preds)  # 使用单位Hessian近似

        return grad, hess

    def fit(self, X, y, eval_set=None):
        # 转换为LightGBM数据集
        train_data = lgb.Dataset(X, label=y)

        # 自定义评估函数
        def trading_metric(preds, train_data):
            y_true = train_data.get_label()
            return 'trading_loss', self.loss_fn(y_true, preds), False

        # 训练模型
        self.model = lgb.train(
            {**self.lgb_params, 'objective': None},
            train_data,
            feval=trading_metric,
            valid_sets=eval_set
        )
        return self

    def predict(self, X):
        return self.model.predict(X)