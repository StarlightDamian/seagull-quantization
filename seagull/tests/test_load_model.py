# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:04:31 2024

@author: awei
"""
#from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Assuming X_train, y_train are your training data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define your LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    # Other parameters...
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# Train the model with early stopping
model = lgb.train(params, train_data, valid_sets=[valid_data], early_stopping_rounds=50, verbose_eval=10)

# Your model is now trained with early stopping

# =============================================================================
# import pandas as pd
# import joblib
# from seagull.settings import PATH
# MULTIOUTPUT_MODEL_PATH = f'{PATH}/checkpoint/lightgbm_regression_short_term_recommend.joblib'
# multioutput_model_path = MULTIOUTPUT_MODEL_PATH
# model_multioutput, model_metadata = joblib.load(multioutput_model_path)
# 
# x_train = pd.read_csv(f'{PATH}/_file/train_short_term_recommend_x_train.csv')
# y_train = pd.read_csv(f'{PATH}/_file/train_short_term_recommend_y_train.csv').values
# model_multioutput.fit(x_train, y_train)
# y_pred = model_multioutput.predict(x_test)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# =============================================================================


#model_multioutput_dict = vars(model_multioutput.estimator)
#print(model_multioutput_dict)
# =============================================================================
# trade_order_details
# import pandas as pd
# 
# # 创建示例DataFrame
# df = pd.DataFrame({
#     'date': pd.date_range('2022-01-01', '2022-01-10'),
#     'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# })
# 
# # 使用agg计算最大值和最小值
# result1, result2 = df['date'].agg(['min', 'max'])
# 
# # 输出结果
# print(result1)
# =============================================================================
