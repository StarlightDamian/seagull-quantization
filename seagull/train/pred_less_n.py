# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:58:46 2024

@author: awei
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# Custom objective function
def custom_asymmetric_objective(y_pred, dataset):
    y_true = dataset.get_label()  # Extract actual values
    residual = y_true - y_pred
    grad = np.where(residual > 0, -0.86 * np.abs(residual), 0.14 * np.abs(residual))  # Larger gradient for bigger errors
    hess = np.ones_like(residual)  # Hessian
    return grad, hess


# Custom evaluation function
def custom_asymmetric_eval(y_pred, dataset):
    y_true = dataset.get_label()  # Extract actual values
    residual = y_true - y_pred
    loss = np.where(residual > 0, 
                    (residual ** 2) * 0.86,  # Heavier penalty when under-predicting
                    (residual ** 2) * 0.14)  # Lighter penalty when over-predicting
    return "custom_asymmetric_eval", np.mean(loss), False


np.random.seed(43)
X = pd.DataFrame(np.random.randn(10000, 10), columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(np.random.uniform(50, 150, 10000), name='num') 
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set parameters
params = {
    'boosting_type': 'gbdt',
    'objective': custom_asymmetric_objective,  # Custom objective
    'metric': 'custom',  # Custom metric
    'num_leaves': 100,  # Increase number of leaves to capture more patterns
    'learning_rate': 0.05,
    'max_depth': 10,  # Add a max depth to prevent overfitting
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


# Train the model
model = lgb.train(params, train_data, num_boost_round=100, 
                  valid_sets=[test_data], 
                  feval=custom_asymmetric_eval,
                  #early_stopping_rounds=10
                  )

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate the proportion of predictions below actual values
below_actual = np.mean(y_pred < y_test)
print(f"Proportion of predictions below actual: {below_actual:.2%}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual num")
plt.ylabel("Predicted num")
plt.title("Actual vs Predicted num")
plt.tight_layout()
plt.show()

# Analyze prediction errors
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# Custom evaluation of predictions
def custom_metric(y_true, y_pred):
    below_actual = np.mean(y_pred < y_true)
    mse = np.mean((y_true - y_pred)**2)
    return below_actual, mse

below_actual, mse = custom_metric(y_test, y_pred)
print(f"Custom Metric - Below Actual: {below_actual:.2%}, MSE: {mse:.4f}")


y1=pd.DataFrame([y_test.values, y_pred]).T
y1.columns=['y_test', 'y_pred']
##(y1.y_test>y1.y_pred).sum()
#Out[13]: 2000