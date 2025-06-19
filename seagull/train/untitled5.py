# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:42:24 2024

@author: awei
"""
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd

from seagull.settings import PATH
from seagull.utils import utils_database, utils_character, utils_log, utils_math
import lightgbm_base
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

# Define the MLP model class in PyTorch
class MLP(nn.Module):
    def __init__(self, num_features, hidden_units, dropout_rates):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(num_features, hidden_units[0])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_units[i]) for i in range(len(hidden_units))])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rates[i]) for i in range(len(dropout_rates))])
        self.dense_layers = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)])
        self.output_layer = nn.Linear(hidden_units[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.batch_norms[0](x)
        x = torch.nn.functional.silu(x)  # Swish activation
        x = self.dropouts[0](x)
        
        for i in range(1, len(self.dense_layers)):
            x = self.dense_layers[i-1](x)
            x = self.batch_norms[i](x)
            x = torch.nn.functional.silu(x)  # Swish activation
            x = self.dropouts[i](x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Parameters
batch_size = 4096
hidden_units = [384, 896, 896, 394]
dropout_rates = [0.10143786981358652, 0.19720339053599725, 0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
label_smoothing = 1e-2
learning_rate = 1e-3

# Load models
num_models = 2
models = []

parser = argparse.ArgumentParser()
parser.add_argument('--date_start', type=str, default='2023-10-20', help='When to start feature engineering')
parser.add_argument('--date_end', type=str, default='2023-12-27', help='End time for feature engineering')
args = parser.parse_args()

logger.info(f"""task: feature_engineering
                date_start: {args.date_start}
                date_end: {args.date_end}""")


# 获取日期段数据
with utils_database.engine_conn("POSTGRES") as conn:
    df = pd.read_sql(f"SELECT * FROM dwd_freq_incr_stock_daily WHERE date BETWEEN '{args.date_start}' AND '{args.date_end}'", con=conn.engine)

features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume',
           'value_traded', 'turnover', 'chg_rel', 'pe_ttm',
          'ps_ttm', 'pcf_ttm', 'pb_mrq','board_type', 'price_limit_rate']

for i in range(num_models):
    model = MLP(num_features=len(features), hidden_units=hidden_units, dropout_rates=dropout_rates)
    #model.load_state_dict(torch.load(f'../input/js-nn-models/JSModel_{i}.pth'))  # Load model weights
    model.eval()  # Set to evaluation mode
    models.append(model)


f_mean = np.random.randn(1, len(features))
test_data = pd.DataFrame({
    'weight': np.random.random(100),  # Random weights for simulation
    'feature_1': np.random.random(100),
    'feature_2': np.random.random(100),
    'feature_3': np.random.random(100),
    # Add more features as needed
})
test_data['action'] = 0  # Placeholder for predictions

# Simulate features from test data (you should adjust according to your features)
features = ['feature_1', 'feature_2', 'feature_3']  # Replace with actual feature names

# Placeholder for the models
models = []  # Assuming the models are already loaded as you did in the original code

opt_th = 0.5  # Optimal threshold for prediction
predictions = []

# Loop over the test data (mockup)
for idx, test_df in tqdm(test_data.iterrows(), total=len(test_data)):
    if test_df['weight'] > 0:
        # Extract features (simulated example)
        x_tt = test_df[features].values

        # Handle missing values
        if np.isnan(x_tt[1:].sum()):  # If there are NaN values
            x_tt[1:] = np.nan_to_num(x_tt[1:]) + np.isnan(x_tt[1:]) * f_mean
        
        # Convert features to tensor
        x_tt_tensor = torch.tensor(x_tt, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Initialize prediction
        pred = 0.0
        
        # Make predictions with the models
        for model in models:
            with torch.no_grad():
                pred += model(x_tt_tensor).item() / len(models)
        
        # Apply threshold to make a decision
        action = 1 if pred >= opt_th else 0
        predictions.append(action)
    else:
        predictions.append(0)  # If weight is zero, set action to 0

# After predictions are made, you can store them in the test_data DataFrame
test_data['action'] = predictions

# For demonstration, let's print the results (you can output to CSV or other formats)
print(test_data[['weight', 'action']])

# In case you need to save the predictions
# test_data.to_csv('predictions.csv', index=False)


        Input  features
         |
         |----------------------------------------
         v                                       |
   +-------------+                               |
   |GaussianNoise|                               |
   +-------------+                               |
         |                                       |
         v                                       |
   +-------------+                               |
   | Encoder     |                               |
   +-------------+                               |
         |                                       |
         |--------------------------------------||
         v                                      ||
   +------------+                               ||
   | Decoder    |                            +-------
   +------------+                            | MLP  |
         | ------------------|               +------+
         v                   |                  |
   +---------------------+   |                  v
   |Classification Layer |   |          Multi-Label Classification
   +---------------------+   |           (BCE Loss) Final Output
         |                   v       
         v              Regression(MSE Loss)         
Multi-Label Classification
(BCE Loss)
         |
         v
   +-----------+
   |   End     |
   +-----------+
