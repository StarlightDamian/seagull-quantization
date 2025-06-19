# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:50:19 2024

@author: awei
demo_optuna
"""
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
#optuna.logging.set_verbosity(optuna.logging.ERROR)

class StockModelManagerOptuna:
    def __init__(self, model_dir='models'):
        self.models = {}  # 保存模型的字典
        self.best_params = {}  # 保存最佳超参数
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)  # 创建模型存储目录

    def train_and_save_models_with_optuna(self, data, target_column, features, full_code_column, n_trials=50):
        """
        分股票训练模型并通过 Optuna 调参
        """
        grouped = data.groupby(full_code_column)
        for full_code, group in grouped:
            print(f"Optimizing model for {full_code}...")
            
            x_train, x_test, y_train, y_test = train_test_split(
                group[features], group[target_column], test_size=0.2, random_state=42
            )
            
            def objective(trial):
                # 定义搜索空间
                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10)
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                return rmse
            
            # 创建 Optuna study 并优化
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            
            # 保存最佳超参数和模型
            self.best_params[full_code] = study.best_params
            print(f"Best RMSE for {full_code}: {study.best_value:.4f}")
            print(f"Best params for {full_code}: {study.best_params}")
            
            # 用最佳参数训练最终模型
            best_model = lgb.LGBMRegressor(**study.best_params)
            best_model.fit(x_train, y_train)
            self.models[full_code] = best_model
            
            # 保存模型到本地
            model_path = os.path.join(self.model_dir, f"{full_code}_model.pkl")
            joblib.dump(best_model, model_path)

    def load_models(self):
        """
        从本地加载所有模型
        """
        for file_name in os.listdir(self.model_dir):
            if file_name.endswith("_model.pkl"):
                full_code = file_name.replace("_model.pkl", "")
                model_path = os.path.join(self.model_dir, file_name)
                self.models[full_code] = joblib.load(model_path)

    def predict(self, full_code, x):
        """
        使用指定股票的模型进行预测
        """
        if full_code not in self.models:
            raise ValueError(f"No model found for full_code: {full_code}")
        model = self.models[full_code]
        return model.predict(x)

# 示例数据
data = pd.DataFrame({
    'full_code': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'TSLA', 'TSLA'],
    'feature1': np.random.rand(6),
    'feature2': np.random.rand(6),
    'target': np.random.rand(6)
})

# 特征和目标列
features = ['feature1', 'feature2']
target_column = 'target'
full_code_column = 'full_code'

# 创建并训练模型管理器
manager = StockModelManagerOptuna(model_dir='stock_models_optuna')
manager.train_and_save_models_with_optuna(data, target_column, features, full_code_column, n_trials=20)

# 加载模型并预测
manager.load_models()
test_data = pd.DataFrame({'feature1': [0.5], 'feature2': [0.6]})
pred = manager.predict('AAPL', test_data)
print(f"Prediction for AAPL: {pred[0]:.4f}")