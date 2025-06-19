# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:34:10 2024

@author: awei
https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp?scriptVersionId=73762661&cellId=2
https://www.kaggle.com/code/backtracking/autoencoder-mlp-cv-multitarget-pytorch-basic
https://www.kaggle.com/code/xiaowangiiiii/current-1th-jane-street-ae-mlp-xgb
https://www.kaggle.com/code/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5/notebook
https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348
https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
https://github.com/flame0409/Jane-Street-Market-Prediction/blob/master/%E7%AE%80%E8%A1%97%E6%95%B0%E6%8D%AE%E9%A2%84%E6%B5%8B%E6%B5%81%E7%A8%8B.md
https://blog.csdn.net/weixin_51484067/article/details/114635812
https://www.kaggle.com/code/gogo827jz/jane-street-neural-network-starter
Jane Street mmlp
mmlp_pytorch

The supervised autoencoder approach was initially proposed in Bottleneck encoder + MLP + Keras Tuner 8601c5, where one supervised autoencoder is trained separately before cross-validation (CV) split. I have realised that this training may cause label leakage because the autoencoder has seen part of the data in the validation set in each CV split and it can generate label-leakage features to overfit. So, my approach is to train the supervised autoencoder along with MLP in one model in each CV split. The training processes and explanations are given in the notebook and the following statements.

Cross-Validation (CV) Strategy and Feature Engineering:

5-fold 31-gap purged group time-series split
Remove first 85 days for training since they have different feature variance
Forward-fill the missing values
Transfer all resp targets (resp, resp_1, resp_2, resp_3, resp_4) to action for multi-label classification
Use the mean of the absolute values of all resp targets as sample weights for training so that the model can focus on capturing samples with large absolute resp.
During inference, the mean of all predicted actions is taken as the final probability
Deep Learning Model:

Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model
Train autoencoder and MLP together in each CV split to prevent data leakage
Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient
Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting
Use swish activation function instead of ReLU to prevent ‘dead neuron’ and smooth the gradient
Batch Normalisation and Dropout are used for MLP
Train the model with 3 different random seeds and take the average to reduce prediction variance
Only use the models (with different seeds) trained in the last two CV splits since they have seen more data
Only monitor the BCE loss of MLP instead of the overall loss for early stopping
Use Hyperopt to find the optimal hyperparameter set
"""
import os

import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from seagull.settings import PATH
from seagull.utils import utils_database, utils_character, utils_log, utils_math
import lightgbm_base
log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{PATH}/log/{log_filename}.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rates, num_class_labels, num_reg_labels):
        super(MMLPModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        #nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')  # He 初始化
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.act1 = nn.ReLU()#nn.SiLU()  # Swish activation

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        #nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        #nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.act3 = nn.ReLU()

        # 解码器部分（重建原始输入）
        self.fc_dec = nn.Linear(hidden_units[2], input_dim)

        # 多任务输出
        self.classification_head = nn.Linear(hidden_units[2], num_class_labels)  # 分类任务输出
        self.regression_head = nn.Linear(hidden_units[2], num_reg_labels)  # 回归任务输出
        
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.dropout3 = nn.Dropout(dropout_rates[2])

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # 解码器部分
        decoder_output = self.fc_dec(x)

        # 分类任务输出
        classification_output = self.classification_head(x)

        # 回归任务输出
        regression_output = self.regression_head(x)

        return decoder_output, classification_output, regression_output, x  # 返回隐层编码 x

def assign_target_class(relative_close):
    # 根据条件分配 target_class
    if relative_close > 0.01:
        return 2
    elif -0.01 > relative_close:
        return 0
    #elif -0.05 <= relative_close <= 0.05:
    else:
        return 1

def onehot(stock_df, label_df):
    # 对分类变量进行编码（如 One-Hot 编码）
    #df = pd.get_dummies(df, drop_first=True)
    # 或者使用 LabelEncoder 对分类标签进行编码（如有序标签）
    #from sklearn.preprocessing import LabelEncoder
    #le = LabelEncoder()
    #df['category_column'] = le.fit_transform(df['category_column'])

    # 示例数据
    #label_df = pd.DataFrame([['001', 'a'], ['001', 'b'], ['002', 'a']], columns=['full_code', 'label'])
    #stock_df = pd.DataFrame({'full_code': ['001', '002', '003']})
    
    # One-Hot 编码
    label_onehot = pd.get_dummies(label_df, columns=['label'], prefix='label_', prefix_sep='')

    # 聚合 One-Hot 编码
    label_onehot_grouped = label_onehot.groupby('full_code').max().reset_index()
    #del label_onehot_grouped['']
    
    # 将 One-Hot 编码结果与 stock_df 合并
    result_df = stock_df.merge(label_onehot_grouped, on='full_code', how='left').fillna(0)#.astype(int)

    #print(result_df)
    return result_df

def custom_loss(y_pred, y_true):
    #y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数，可以调整

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, -alpha * np.sign(delta))
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)
    return grad, hess

if __name__ == '__main__':
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
        label_df = pd.read_sql("dwd_tags_full_label", con=conn.engine)
        macd_df = pd.read_sql(f"""
                    SELECT 
                        primary_key
                        ,volume_slope_12_26_9
                        ,value_traded_slope_12_26_9
                        ,turnover_slope_12_26_9
                        ,volume_hist_12_26_9
                        ,value_traded_hist_12_26_9
                        ,turnover_hist_12_26_9
                        ,volume_diff_1
                        ,volume_diff_5
                        ,volume_diff_30
                        ,value_traded_diff_1
                        ,value_traded_diff_5
                        ,value_traded_diff_30
                        ,value_traded_hist_diff_1
                        ,volume_hist_diff_1
                    FROM 
                        dwd_feat_incr_macd
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'""", con=conn.engine)  # max_date=2024-08-14
        flow_df = pd.read_sql(f"""
                    SELECT
                        primary_key
                        ,log10_main_inflow
                        ,log10_ultra_large_inflow
                        ,log10_large_inflow
                        ,log10_medium_inflow
                        ,log10_small_inflow
                    FROM
                        dwd_feat_incr_capital_flow_temp
                    WHERE
                        date BETWEEN '{args.date_start}' AND '{args.date_end}'""", con=conn.engine)  # 2021-08-25, 2024-10-31
    
    # 示例数据
# =============================================================================
#     data = pd.DataFrame({
#         'ticker': ['AAPL', 'GOOGL', 'TSLA', 'AMC', 'XYZ'] * 100,
#         'market_cap': np.random.rand(500) * 1e12,
#         'revenue_growth': np.random.rand(500),
#         'profit_growth': np.random.rand(500),
#         'dividend_yield': np.random.rand(500),
#         'sector': np.random.choice(['tech', 'auto', 'finance', 'healthcare'], size=500),
#         'target_class': np.random.choice([0, 1, 2], size=500),
#         'target_value': np.random.rand(500) * 100
#     })
# =============================================================================

    #df = pd.read_csv(f'{PATH}/data/test_dwd_freq_incr_stock_daily.csv', low_memory=False)
    df[['prev_close']] = df[['close']].shift(1)
    df[['high_rate', 'low_rate', 'open_rate', 'close_rate','avg_price_rate']] = df[['high', 'low', 'open', 'close','avg_price']].div(df['prev_close'], axis=0)
    df[['next_high_rate', 'next_low_rate', 'next_close_rate']] = df[['high_rate', 'low_rate', 'close_rate']].shift(-1)
    df = df.head(-1)
    
    index_df = df.loc[df['full_code']=='000001.sh', ['date', 'next_close_rate']]
    index_df = index_df.rename(columns={'next_close_rate': 'next_index_close_rate'})
    
    df = pd.merge(df, index_df, on='date', how='left')
    df['next_relative_close_rel'] = df['next_close_rate'] - df['next_index_close_rate']
    df['target_class'] = df['next_relative_close_rel'].apply(assign_target_class)
    # df['target_class'].value_counts()
    
    # 指定需要loge的列
    columns_to_apply = ['volume','value_traded','pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
    df[columns_to_apply] = df[columns_to_apply].applymap(utils_math.log_e)

# =============================================================================
#     # 多标签进行onehot， 先置空加快计算
#     df = onehot(df, label_df[['full_code', 'label']])
#     label_list =  [x for x in df.columns if ('label_' in x) and ('label_昨日' not in x)]#[]#
# =============================================================================
    label_list = []
    
    df = pd.merge(df, macd_df, on='primary_key', how='left')
    macd_list = ['volume_slope_12_26_9', 'value_traded_slope_12_26_9',
           'turnover_slope_12_26_9', 'volume_hist_12_26_9',
           'value_traded_hist_12_26_9', 'turnover_hist_12_26_9', 'volume_diff_1',
           'volume_diff_5', 'volume_diff_30', 'value_traded_diff_1',
           'value_traded_diff_5', 'value_traded_diff_30',
           'value_traded_hist_diff_1', 'volume_hist_diff_1']
    
    df = pd.merge(df, flow_df, on='primary_key', how='left')
    flow_list = ['log10_main_inflow', 'log10_ultra_large_inflow',
           'log10_large_inflow', 'log10_medium_inflow', 'log10_small_inflow']
    # 清洗脏数据
    df = df[(df.high_rate-1<=df.price_limit_rate)&(df.low_rate-1>=-df.price_limit_rate)]
    
    # 特征和目标
    x = df[['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume',
               'value_traded', 'turnover', 'chg_rel', 'pe_ttm',
              'ps_ttm', 'pcf_ttm', 'pb_mrq', 'price_limit_rate']+label_list+macd_list+flow_list]#,'board_type'
    y_class = df['target_class']
    y_reg = df['next_high_rate']
    
    # 处理异常值
    x_train = x.replace([np.inf, -np.inf, np.nan], 1)#.fillna(1)  # debug:这个错误是由于输入数据中包含无穷值 (inf)、负无穷值 (-inf)、或超出 float64 类型范围的值导致的
    y_class_train = y_class.replace([np.inf, -np.inf, np.nan], 0)#.fillna(0)
    y_reg_train = y_reg.replace([np.inf, -np.inf, np.nan], 1)#.fillna(1)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    y_reg_train_min = y_reg_train.min()
    y_reg_train_max = y_reg_train.max()
    y_reg_train = (y_reg_train - y_reg_train_min) / (y_reg_train_max - y_reg_train_min)
    # 假设训练数据
    #x_train = np.random.rand(1000, 130)  # 1000 个样本，130 个特征
    #y_class_train = np.random.choice([0, 1, 2, 3, 4], size=1000)  # 每个样本 1 个标签，类别为 [0, 1, 2, 3, 4]
    #y_reg_train = np.random.rand(1000, 3)  # 假设回归目标为 5 个标签

    x_train = x_train.apply(pd.to_numeric, errors='coerce')  # 将无法转换的非数值数据转换为 NaN
    y_class_train = y_class_train.apply(pd.to_numeric, errors='coerce')
    y_reg_train = y_reg_train.apply(pd.to_numeric, errors='coerce')
    x_train = scaler.fit_transform(x_train)  # 标准化输入数据
    # 转换为 Tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_class_train_tensor = torch.tensor(y_class_train.values, dtype=torch.long).to(device)  # 使用 long 类型作为标签
    y_reg_train_tensor = torch.tensor(y_reg_train.values, dtype=torch.float32).to(device)  # 回归标签

    # 创建 DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_class_train_tensor, y_reg_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    
    
    # 模型实例化
    input_dim = x_train.shape[1]  #130  # 输入特征维度
    hidden_units = [128, 64, 32]  # 隐藏层单元,64
    dropout_rates = [0.1, 0.1, 0.1]  # dropout比率
    num_class_labels = 3  # y_class_train.shape[1]  # 分类标签数量
    num_reg_labels = 1  # y_reg_train.shape[1]
    model = MMLPModel(input_dim, hidden_units, dropout_rates, num_class_labels, num_reg_labels).to(device)
    # 损失函数和优化器
    criterion_class = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    criterion_reg = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 5
    # 给分类损失增加权重
    classification_loss_weight = 1.0
    regression_loss_weight = 10.0
    for epoch in range(epochs):
        model.train()
        running_loss_class = 0.0
        running_loss_reg = 0.0
        
        for i, (x_batch, y_class_batch, y_reg_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            decoder_output, class_output, reg_output, hidden_encoding = model(x_batch)
    
            # 计算损失
            loss_class = criterion_class(class_output.squeeze(), y_class_batch)
            loss_reg = criterion_reg(reg_output.squeeze(), y_reg_batch)
    
            # 总损失
            #total_loss = loss_class + loss_reg
            total_loss = classification_loss_weight * loss_class + regression_loss_weight * loss_reg
            total_loss.backward()
    
            optimizer.step()
    
            running_loss_class += loss_class.item()
            running_loss_reg += loss_reg.item()
    
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Classification Loss: {running_loss_class / len(train_loader):.4f}, "
              f"Regression Loss: {running_loss_reg / len(train_loader):.4f}")
    # 假设你已经获得了隐层编码作为特征，y_class_train 是标签
    # 将多分类标签（0, 1, 2, 3, 4）作为目标
    #X_train = np.random.rand(1000, 128)  # 假设通过 MMLP 提取的隐层编码（128维）
    #y_class_train = np.random.choice([0, 1, 2, 3, 4], size=1000)  # 每个样本 1 个标签（类别0,1,2,3,4）
    # 提取隐层编码（hidden_encoding）
    model.eval()
    with torch.no_grad():
        hidden_encoding = []
        for x_batch, _, _ in train_loader:
            _, class_output, reg_output, encoding = model(x_batch)
            hidden_encoding.append(encoding)  # 提取隐层编码作为特征
        
        hidden_encoding = torch.cat(hidden_encoding, dim=0)
        #hidden_encoding = hidden_encoding.numpy()  # 转换为 numpy 数组
        
    # 拆分数据集
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_class_train, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(hidden_encoding, y_class_train, test_size=0.2, random_state=42)

    # 创建 LightGBM 数据集
    X_train = X_train.cpu().numpy()
    X_val = X_val.cpu().numpy()
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train) 

    # LightGBM 参数
    params = {
        'objective': 'multiclass',  # 多类分类
        'num_class': 5,  # 5 个类别
        'metric': 'multi_logloss',  # 多类损失
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    # 训练模型
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_val])

    # 预测
    y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)# y_pred.shape (127168, 5)

    # 计算 ROC AUC 分数
    auc_score = roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')
    print("ROC AUC Score:", auc_score)
    
    #将价量因子先输入深度学习模型，取深度学习模型最后的隐层编码作为特征输入到树模型中，将两模型串行连接。

# =============================================================================
# # 测试模型性能
# model.eval()
# with torch.no_grad():
#     # 假设有验证数据
#     x_val = torch.rand(200, input_dim, dtype=torch.float32).to(device)
#     y_class_val = torch.randint(0, num_labels, (200,), dtype=torch.long).to(device)
#     y_reg_val = torch.rand(200, num_labels, dtype=torch.float32).to(device)
# 
#     _, class_output, reg_output, _ = model(x_val)
#     val_loss_class = criterion_class(class_output, y_class_val).item()
#     val_loss_reg = criterion_reg(reg_output, y_reg_val).item()
# 
# print(f"Validation - Classification Loss: {val_loss_class:.4f}, Regression Loss: {val_loss_reg:.4f}")
# 
# =============================================================================
