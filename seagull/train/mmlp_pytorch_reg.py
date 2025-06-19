# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:14:50 2025

@author: awei
(mmlp_pytorch_reg)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler

from seagull.settings import PATH
from seagull.utils import utils_database, utils_log, utils_thread, utils_math

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)
import numpy as np



def rmse(y_true, y_pred):
    return (torch.mean((y_true - y_pred) ** 2)) ** 0.5

class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_labels):
        super(MMLPModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.act1 = nn.SiLU()  # Swish activation

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.act2 = nn.SiLU()

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.act3 = nn.SiLU()

        self.fc_dec = nn.Linear(hidden_units[2], input_dim)  # Decoder output

        # 回归任务头
        self.regression_head = nn.Linear(hidden_units[2], num_labels)  # 回归任务

        self.dropout = nn.Dropout(dropout_rates[0])

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # 解码器部分
        # decoder_output = self.fc_dec(x)

        # 回归任务输出
        regression_output = self.regression_head(x)  # 输出 [batch_size, num_pred]

        return regression_output  

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
if __name__ == '__main__':
    #raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train.feather')
    raw_df = pd.read_feather(f'{PATH}/data/das_wide_incr_train_20230103_20241220.feather')
    # 清洗脏数据
    raw_df = raw_df[(raw_df.high <= raw_df.limit_up)&(raw_df.low >= raw_df.limit_down)]
    raw_df = raw_df[(raw_df.high <= raw_df.limit_up)&
                        (raw_df.low >= raw_df.limit_down)&
                        (raw_df.next_high_rate-1<raw_df.price_limit_rate)&
                        (1-raw_df.next_low_rate<raw_df.price_limit_rate)
                        ]
    raw_df = raw_df[~((raw_df.next_high_rate.apply(np.isinf))|(raw_df.next_low_rate.apply(np.isinf)))]
    
    ohlc_features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume', 'turnover', 'turnover_pct',
                     'price_limit_rate', 'date_diff_prev', 'date_diff_next','is_limit_down_prev',
                     'is_limit_up_prev']#,'board_type', 'date_week'
    fundamental_features = ['chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
    # =============================================================================
    #     # 多标签进行onehot， 先置空加快计算
    #     train_df = onehot(train_df, label_df[['full_code', 'label']])
    #     label_features =  [x for x in train_df.columns if ('label_' in x) and ('label_昨日' not in x)]#[]#
    # =============================================================================
    label_features = []
    macd_features = ['close_slope_12_26_9'
                ,'volume_slope_12_26_9'
                ,'turnover_slope_12_26_9'
                ,'turnover_pct_slope_12_26_9'
                ,'close_acceleration_12_26_9'
                ,'volume_acceleration_12_26_9'
                ,'turnover_acceleration_12_26_9'
                ,'turnover_pct_acceleration_12_26_9'
                ,'close_hist_12_26_9'
                ,'volume_hist_12_26_9'
                ,'turnover_hist_12_26_9'
                ,'turnover_pct_hist_12_26_9'
                ,'close_diff_1'
                ,'close_diff_5'
                ,'close_diff_30'
                ,'volume_diff_1'
                ,'volume_diff_5'
                ,'volume_diff_30'
                ,'turnover_diff_1'
                ,'turnover_diff_5'
                ,'turnover_diff_30'
                ,'turnover_hist_diff_1'
                ,'volume_hist_diff_1'
                ,'close_hist_diff_1']
    flow_features = ['loge_main_inflow', 'loge_ultra_large_inflow',
           'loge_large_inflow', 'loge_medium_inflow', 'loge_small_inflow','loge_main_small_net_inflow',
           'main_inflow_slope_12_26_9', 'ultra_large_inflow_slope_12_26_9',
           'large_inflow_slope_12_26_9', 'medium_inflow_slope_12_26_9',
           'small_inflow_slope_12_26_9', 'main_inflow_acceleration_12_26_9',
           'ultra_large_inflow_acceleration_12_26_9',
           'large_inflow_acceleration_12_26_9',
           'medium_inflow_acceleration_12_26_9',
           'small_inflow_acceleration_12_26_9', 'main_inflow_hist_12_26_9',
           'ultra_large_inflow_hist_12_26_9', 'large_inflow_hist_12_26_9',
           'medium_inflow_hist_12_26_9', 'small_inflow_hist_12_26_9',
           'main_inflow_diff_1', 'main_inflow_diff_5', 'main_inflow_diff_30',
           'ultra_large_inflow_diff_1', 'ultra_large_inflow_diff_5',
           'ultra_large_inflow_diff_30', 'large_inflow_diff_1',
           'large_inflow_diff_5', 'large_inflow_diff_30', 'medium_inflow_diff_1',
           'medium_inflow_diff_5', 'medium_inflow_diff_30', 'small_inflow_diff_1',
           'small_inflow_diff_5', 'small_inflow_diff_30',
           'main_inflow_hist_diff_1', 'ultra_large_inflow_hist_diff_1',
           'large_inflow_hist_diff_1', 'medium_inflow_hist_diff_1',
           'small_inflow_hist_d-0iff_1',
           'main_small_net_inflow_slope_12_26_9',
           'main_small_net_inflow_acceleration_12_26_9',
           'main_small_net_inflow_hist_12_26_9',
           'main_small_net_inflow_diff_1',
           'main_small_net_inflow_diff_5',
           'main_small_net_inflow_diff_30',
           'main_small_net_inflow_hist_diff_1']
    alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']
    index_features = ['index_close_diff_1', 'index_close_diff_5', 'index_close_diff_30', 'index_volume_diff_1','index_volume_diff_5',
                      'index_volume_diff_30', 'index_turnover_diff_1', 'index_turnover_diff_5', 'index_turnover_diff_30','index_close_rate',
                      'index_close','index_volume','index_turnover','index_turnover_pct']
    indicators_features = ['rsi','cci', 'wr', 'vwap', 'ad', 'mom', 'atr', 'adx', 'plus_di', 'minus_di', 'mfi', 'upper_band', 'middle_band', 'lower_band', 'kdj_fastk','kdj_fastd']
    #'board_type','date_week',
    categorical_features = ['is_limit_down_prev','is_limit_up_prev','alpha061','alpha062','alpha064','alpha065','alpha068',
                            'alpha074','alpha075','alpha081','alpha086','alpha095','alpha099']
    #alpha068
    features = indicators_features +\
               macd_features +\
               alpha_features +\
               ohlc_features +\
               fundamental_features +\
               index_features
               # label_features +\
               # flow_features +\
                              
    # 假设训练数据
    #x_train = np.random.rand(1000, 130)  # 1000 个样本，130 个特征
    #y_train = np.random.rand(1000, 5)    # 5 个回归任务标签
    raw_df[categorical_features] = raw_df[categorical_features].astype(float)#.astype('bool')
    
    raw_df.sort_values(by='date', ascending=True, inplace=True, ignore_index=True)
    
    # 特征和目标
    TARGET_NAMES = ['next_high_rate',
                    'next_low_rate',
                    'next_close_rate',
                    'y_10d_vwap_rate',
                    'y_10d_max_dd',
                    'y_10d_high_rate',
                    'y_10d_low_rate',
                    ]
    raw_x = raw_df.loc[raw_df.date<='2024-12-02', features]  # raw_df[features+['date']]#
    y = raw_df.loc[raw_df.date<='2024-12-02', TARGET_NAMES]  # raw_df[['next_based_index_class','date']]#
    X_val = raw_df.loc[raw_df.date>'2024-12-03', features]
    y_val = raw_df.loc[raw_df.date>'2024-12-03', TARGET_NAMES]
    
    # 滚动训练和交叉验证
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 模型实例化
    input_dim = len(features)  # 输入特征维度
    hidden_units = [128, 64, 32]  # 隐藏层单元
    dropout_rates = [0.0]  # dropout比率
    num_pred = len(TARGET_NAMES)  # 回归任务的标签数量（即预测值的维度）
    
    model = MMLPModel(input_dim, hidden_units, dropout_rates, num_pred).to(device)
    model.apply(init_weights) # 初始化权重
    
    # 损失函数和优化器
    criterion_reg = nn.MSELoss()  # 回归任务的均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    # 转换为 Tensor

    
    # 拟合并转化训练集
    x = np.nan_to_num(raw_x, nan=0.0)
    #x = np.nan_to_num(x, posinf=1e10, neginf=-1e10)
    x = np.nan_to_num(x, posinf=1, neginf=-1)
    
    # 训练集和验证集的标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)
    x_tensor = torch.nan_to_num(x_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    for train_index, test_index in tscv.split(x_tensor):
        
        x_train_tensor, x_test_tensor = x_tensor[train_index], x_tensor[test_index]
        y_train_tensor, y_test_tensor = y_tensor[train_index], y_tensor[test_index]
        
        logger.info(f'{x_train_tensor.shape[0]} {train_index.max()}')
        logger.info(f'{y_train_tensor.shape[0]} {test_index.min()}')
        
        # 创建 DataLoader
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  #, pin_memory=True
        
        #test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        # 训练模型
        epochs = 5
        for epoch in range(epochs):
            model.train()
            running_loss_reg = 0.0
            for idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
        
                # 前向传播
                y_output = model(x_batch)
        
                # 计算损失
                loss_reg = criterion_reg(y_output, y_batch)

                # 反向传播
                loss_reg.backward()
                optimizer.step()
                
                running_loss_reg += loss_reg.item()
        
            logger.info(f"Epoch {epoch+1}/{epochs}, Regression Loss: {running_loss_reg / len(train_loader):.7f}")
            
            # 测试集
# =============================================================================
#             y_test_pred = model(x_test_tensor)
#             rmse_values = {target: rmse(y_test_tensor[:, idx], y_test_pred[:, idx]).item() 
#                            for idx, target in enumerate(TARGET_NAMES)}
#             for key, value in rmse_values.items():
#                 print(f"{key:<{15}} : {value:>{12}.6f}")
# =============================================================================
            #  torch.cuda.empty_cache()  # debug: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.26 GiB. GPU 0 has a total capacty of 6.00 GiB of which 0 bytes is free. Of the allocated memory 3.10 GiB is allocated by PyTorch, and 1.17 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

    x_val = np.nan_to_num(X_val, nan=0.0)
    x_val = np.nan_to_num(x_val, posinf=1e10, neginf=-1e10)
    x_val_scaled = scaler.transform(x_val)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32).to(device)
    x_val_tensor = torch.nan_to_num(x_val_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    y_val_pred_tensor = model(x_val_tensor)
    y_val_pred = y_val_pred_tensor.cpu().detach().numpy()
    #y_val_pred_series = pd.Series(y_val_pred.flatten(), index=y_val.index, name='predicted_next_high_rate')
    #result_df = pd.concat([y_val, y_val_pred_series], axis=1)
    #result_df.to_csv(f'{PATH}/data/mmlp_reg.csv',index=False)
    
    y_val_pred = pd.DataFrame(y_val_pred,columns=TARGET_NAMES)
    y_val = y_val.reset_index(drop=True)
    rmse_values = y_val.apply(lambda col: rmse(col, y_val_pred[col.name]))

    #y_val_pred_tensor, y_val

    #rmse_values = y_true.apply(lambda col: rmse(col, y_pred[col]))
    #print(rmse_values)
# =============================================================================
# rmse_values
# Out[22]: 
# next_high_rate     0.036750
# next_low_rate      0.023146
# next_close_rate    0.037686
# y_10d_vwap_rate    0.062314
# y_10d_max_dd       0.066500
# y_10d_high_rate    0.077672
# y_10d_low_rate     0.070304
# 
# =============================================================================
# =============================================================================test
# next_high_rate  :     0.016680
# next_low_rate   :     0.014035
# next_close_rate :     0.017670
# y_10d_vwap_rate :     0.037181
# y_10d_max_dd    :     0.029135
# y_10d_high_rate :     0.057868
# y_10d_low_rate  :     0.027207
# =============================================================================

# =============================================================================val
# next_high_rate     0.049000
# next_low_rate      0.046066
# next_close_rate    0.049096
# y_10d_vwap_rate    0.075679
# y_10d_max_dd       0.070092
# y_10d_high_rate    0.097066
# y_10d_low_rate     0.062813
# =============================================================================
# =============================================================================
# y_val.apply(lambda col: rmse(col, y_val_pred[col.name]))
# Out[4]: 
# next_high_rate     0.054719
# next_low_rate      0.049241
# next_close_rate    0.054256
# y_10d_vwap_rate    0.082830
# y_10d_max_dd       0.079715
# y_10d_high_rate    0.117000
# y_10d_low_rate     0.069628
# dtype: float64
# =============================================================================
# =============================================================================
# Epoch 4/10, Regression Loss: 0.000324
# Epoch 5/10, Regression Loss: 0.000322
# Epoch 6/10, Regression Loss: 0.000321
# Epoch 7/10, Regression Loss: 0.000319
# Epoch 8/10, Regression Loss: 0.000318
# Epoch 9/10, Regression Loss: 0.000317
# Epoch 10/10, Regression Loss: 0.000316
# 2025-01-15 01:50:29.573 | INFO     | __main__:<module>:214 - 1480752 1480751
# 2025-01-15 01:50:29.575 | INFO     | __main__:<module>:215 - 1480752 1480752
# Epoch 1/10, Regression Loss: 0.000328
# Epoch 2/10, Regression Loss: 0.000321
# Epoch 3/10, Regression Loss: 0.000318
# Epoch 4/10, Regression Loss: 0.000316
# Epoch 5/10, Regression Loss: 0.000314
# Epoch 6/10, Regression Loss: 0.000313
# Epoch 7/10, Regression Loss: 0.000311
# Epoch 8/10, Regression Loss: 0.000311
# Epoch 9/10, Regression Loss: 0.000309
# Epoch 10/10, Regression Loss: 0.000308
# =============================================================================
# =============================================================================
# Epoch 7/20, Regression Loss: 0.000327
# Epoch 8/20, Regression Loss: 0.000325
# Epoch 9/20, Regression Loss: 0.000324
# Epoch 10/20, Regression Loss: 0.000322
# Epoch 11/20, Regression Loss: 0.000321
# Epoch 12/20, Regression Loss: 0.000320
# Epoch 13/20, Regression Loss: 0.000318
# Epoch 14/20, Regression Loss: 0.000317
# Epoch 15/20, Regression Loss: 0.000315
# Epoch 16/20, Regression Loss: 0.000314
# Epoch 17/20, Regression Loss: 0.000313
# Epoch 18/20, Regression Loss: 0.000312
# Epoch 19/20, Regression Loss: 0.000311
# Epoch 20/20, Regression Loss: 0.000310
# 2025-01-15 07:22:11.474 | INFO     | __main__:<module>:214 - 1974336 1974335
# 2025-01-15 07:22:11.476 | INFO     | __main__:<module>:215 - 1974336 1974336
# Epoch 1/20, Regression Loss: 0.000356
# Epoch 2/20, Regression Loss: 0.000343
# Epoch 3/20, Regression Loss: 0.000338
# Epoch 4/20, Regression Loss: 0.000335
# Epoch 5/20, Regression Loss: 0.000333
# Epoch 6/20, Regression Loss: 0.000331
# Epoch 7/20, Regression Loss: 0.000330
# Epoch 8/20, Regression Loss: 0.000328
# Epoch 9/20, Regression Loss: 0.000327
# Epoch 10/20, Regression Loss: 0.000326
# Epoch 11/20, Regression Loss: 0.000325
# Epoch 12/20, Regression Loss: 0.000324
# Epoch 13/20, Regression Loss: 0.000323
# Epoch 14/20, Regression Loss: 0.000322
# Epoch 15/20, Regression Loss: 0.000321
# Epoch 16/20, Regression Loss: 0.000320
# Epoch 17/20, Regression Loss: 0.000320
# Epoch 18/20, Regression Loss: 0.000319
# Epoch 19/20, Regression Loss: 0.000319
# Epoch 20/20, Regression Loss: 0.000318
# C:\Users\awei\AppData\Roaming\Python\Python311\site-packages\numpy\core\fromnumeric.py:88: RuntimeWarning: invalid value encountered in reduce
#   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
# =============================================================================

# =============================================================================
# Epoch 32/50, Regression Loss: 0.000316
# Epoch 33/50, Regression Loss: 0.000315
# Epoch 34/50, Regression Loss: 0.000315
# Epoch 35/50, Regression Loss: 0.000315
# Epoch 36/50, Regression Loss: 0.000314
# Epoch 37/50, Regression Loss: 0.000314
# Epoch 38/50, Regression Loss: 0.000314
# Epoch 39/50, Regression Loss: 0.000313
# Epoch 40/50, Regression Loss: 0.000313
# Epoch 41/50, Regression Loss: 0.000313
# Epoch 42/50, Regression Loss: 0.000312
# Epoch 43/50, Regression Loss: 0.000312
# Epoch 44/50, Regression Loss: 0.000312
# Epoch 45/50, Regression Loss: 0.000312
# Epoch 46/50, Regression Loss: 0.000312
# Epoch 47/50, Regression Loss: 0.000311
# Epoch 48/50, Regression Loss: 0.000311
# Epoch 49/50, Regression Loss: 0.000311
# Epoch 50/50, Regression Loss: 0.000311
# C:\Users\awei\AppData\Roaming\Python\Python311\site-packages\numpy\core\fromnumeric.py:88: RuntimeWarning: invalid value encountered in reduce
#   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
# 
# 
# =============================================================================
# =============================================================================
#     x_train_tensor, x_test_tensor = x_tensor[train_index], x_tensor[test_index]
# 
# OutOfMemoryError: CUDA out of memory. Tried to allocate 1.26 GiB. GPU 0 has a total capacty of 6.00 GiB of which 0 bytes is free. Of the allocated memory 3.10 GiB is allocated by PyTorch, and 1.17 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# 
# =============================================================================
