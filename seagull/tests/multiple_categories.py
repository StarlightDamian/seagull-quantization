# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:14:03 2023

@author: awei
"""
import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
y = np.column_stack((y1, y2))  # Stack the targets horizontally

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM Regressor model
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 10,
    'learning_rate': 0.05,
}
lgb_regressor = lgb.LGBMRegressor(**params)

# Create a MultiOutputRegressor with the LightGBM model
multioutput_regressor = MultiOutputRegressor(lgb_regressor)

# Fit the model to the training data
multioutput_regressor.fit(X_train, y_train)

# =============================================================================
# # Save each LightGBM model to TXT
# path = 'E:/03_software_engineering/github/quantitative-finance/checkpoint'
# for i, estimator in enumerate(multioutput_regressor.estimators_):
#     model_path = f'{path}/model_{i}.txt'
#     estimator.booster_.save_model(model_path)
# 
# # Load each LightGBM model back
# loaded_estimators = []
# for i in range(len(multioutput_regressor.estimators_)):
#     model_path = f'{path}/model_{i}.txt'
#     loaded_model = lgb.Booster(model_file=model_path)
#     loaded_estimators.append(loaded_model)
# 
# # Replace the estimators in the MultiOutputRegressor
# multioutput_regressor.estimators_ = loaded_estimators
# 
# # Make predictions on the test data
# y_pred = multioutput_regressor.predict(X_test)
# 
# # Evaluate the performance
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# =============================================================================
path = 'E:/03_software_engineering/github/quantitative-finance'

import joblib

# Save model and metadata
model_metadata = {'feature_names': ['amount', 'close', 'date_diff', 'date_week_Friday', 'date_week_Monday']}
joblib.dump((multioutput_regressor, model_metadata), f'{path}/checkpoint/multioutput_test_regressor_model.joblib')

# Load model and metadata
multioutput_regressor1, loaded_metadata = joblib.load(f'{path}/checkpoint/multioutput_test_regressor_model.joblib')

# =============================================================================
# from sklearn.multioutput import MultiOutputRegressor
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np
# 
# # Generate some sample data
# np.random.seed(42)
# X = np.random.rand(100, 5)  # 100 samples, 5 features
# y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
# y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
# y = np.column_stack((y1, y2))  # Stack the targets horizontally
# 
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Create a LightGBM Regressor model
# params = {
#     'task': 'train',
#     'boosting': 'gbdt',
#     'objective': 'regression',
#     'num_leaves': 10,
#     'learning_rate': 0.05,
#     'metric': ['mae'],
#     'verbose': -1,
# }
# lgb_regressor = lgb.LGBMRegressor(**params)
# 
# # Create a MultiOutputRegressor with the LightGBM model
# multioutput_regressor = MultiOutputRegressor(lgb_regressor)
# 
# # Fit the model to the training data
# multioutput_regressor.fit(X_train, y_train)
# 
# import joblib
# 
# # Save the MultiOutputRegressor to a file
# path = 'E:/03_software_engineering/github/quantitative-finance'
# MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/multioutput_regressor_model.joblib'
# joblib.dump(multioutput_regressor, MULTIOUTPUT_MODEL_PATH)
# 
# # Load the MultiOutputRegressor back from the file
# loaded_multioutput_regressor = joblib.load(MULTIOUTPUT_MODEL_PATH)
# 
# # Make predictions on the training set using the loaded model
# y_pred_loaded = loaded_multioutput_regressor.predict(X_train)
# 
# # Compare the predictions
# #mse = mean_squared_error(y_test, y_pred)
# mse_loaded = mean_squared_error(y_train, y_pred_loaded)
# #print(f'Mean Squared Error (Original Model): {mse}')
# print(f'Mean Squared Error (Loaded Model): {mse_loaded}')
# 
# import joblib
# 
# # Save model and metadata
# model_metadata = {'feature_names': feature_names}
# joblib.dump((model, model_metadata), 'model_with_metadata.joblib')
# 
# # Load model and metadata
# loaded_model, loaded_metadata = joblib.load('model_with_metadata.joblib')
# =============================================================================

# =============================================================================
# # Save the model
# path = 'E:/03_software_engineering/github/quantitative-finance/checkpoint'
# LGBM_MODEL_PATH = f'{path}/test.txt'
# multioutput_regressor.estimators_[0].booster_.save_model(LGBM_MODEL_PATH)
# 
# # Load the model back
# loaded_model = lgb.Booster(model_file=LGBM_MODEL_PATH)
# multioutput_regressor.estimators_ = [loaded_model]
# 
# # Create a MultiOutputRegressor with the loaded LightGBM model
# #multioutput_regressor_loaded = MultiOutputRegressor(lgb.LGBMRegressor())
# #multioutput_regressor_loaded.estimators_ = [loaded_model]
# # Make predictions on the test data
# print(X_test.shape)
# y_pred = multioutput_regressor.predict(X_test)
# 
# # Evaluate the performance
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# 
# =============================================================================
# =============================================================================
# from sklearn.multioutput import MultiOutputRegressor
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np
# #from sklearn.linear_model import LinearRegression
# # Generate some sample data
# np.random.seed(42)
# X = np.random.rand(100, 5)  # 100 samples, 5 features
# y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
# y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
# y = np.column_stack((y1, y2))  # Stack the targets horizontally
# 
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Create a LightGBM Regressor model
# params = {
#     'task': 'train',
#     'boosting': 'gbdt',
#     'objective': 'regression',
#     'num_leaves': 10, # 决策树上的叶子节点的数量，控制树的复杂度
#     'learning_rate': 0.05,
#     'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error'
#     'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
# }
# lgb_regressor = lgb.LGBMRegressor(**params)
# #base_regressor = LinearRegression()
# # Create a MultiOutputRegressor with the LightGBM model
# multioutput_regressor = MultiOutputRegressor(lgb_regressor)
# 
# # Fit the model to the training data
# multioutput_regressor.fit(X_train, y_train)
# path='E:\\03_software_engineering\\github\\quantitative-finance'
# LGBM_MODEL_PATH = f'{path}/checkpoint/test.txt'
# multioutput_regressor.estimators_[0].booster_.save_model(LGBM_MODEL_PATH)
# multioutput_regressor = lgb.Booster(model_file=LGBM_MODEL_PATH)
# # Make predictions on the test data
# y_pred = multioutput_regressor.predict(X_test)
# 
# # Evaluate the performance
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# =============================================================================

# =============================================================================
# from sklearn.multioutput import MultiOutputRegressor
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np
# 
# # Generate some sample data
# np.random.seed(42)
# X = np.random.rand(100, 5)  # 100 samples, 5 features
# y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
# y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
# y = np.column_stack((y1, y2))  # Stack the targets horizontally
# 
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Create a LightGBM Regressor model
# #base_regressor = LGBMRegressor()
# params = {
#     'task': 'train',
#     'boosting': 'gbdt',
#     'objective': 'regression',
#     'num_leaves': 10, # 决策树上的叶子节点的数量，控制树的复杂度
#     'learning_rate': 0.05,
#     'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error'
#     'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
# }
# lgb_train = lgb.Dataset(X_train, y_train)
# model = lgb.train(params, train_set=lgb_train)
# 
# # Create a MultiOutputRegressor with the LightGBM model
# multioutput_regressor = MultiOutputRegressor(model)
# 
# # Fit the model to the training data
# multioutput_regressor.fit(X_train, y_train)
# 
# # Make predictions on the test data
# y_pred = multioutput_regressor.predict(X_test)
# 
# # Evaluate the performance
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# =============================================================================


# =============================================================================
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np
# 
# # Generate some sample data
# np.random.seed(42)
# X = np.random.rand(100, 5)  # 100 samples, 5 features
# y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
# y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
# y = np.column_stack((y1, y2))  # Stack the targets horizontally
# 
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Create a Linear Regression model
# base_regressor = LinearRegression()
# 
# # Create a MultiOutputRegressor with the base model
# multioutput_regressor = MultiOutputRegressor(base_regressor)
# 
# # Fit the model to the training data
# multioutput_regressor.fit(X_train, y_train)
# 
# # Make predictions on the test data
# y_pred = multioutput_regressor.predict(X_test)
# 
# # Evaluate the performance
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# 
# =============================================================================
