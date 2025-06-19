# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:47:04 2024

@author: awei
价格模型示例(demo_train_2_stock_price)
"""
import lightgbm as lgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from seagull.settings import PATH
# 定义自定义损失函数
# 自定义损失函数
def custom_loss(y_pred, dataset):
    y_true = dataset.get_label()
    residual = (y_true - y_pred).astype(float)
    
    # 定义损失函数参数
    alpha = 10.0  # 控制惩罚力度
    beta = 0.966    # 控制损失函数的非对称性
    
    # 计算梯度
    grad = np.where(residual >= 0, 
                    -2 * alpha * (1 - beta) * residual,
                    -2 * alpha * beta * residual)
    
    # 计算二阶导数
    hess = np.where(residual >= 0,
                    2 * alpha * (1 - beta),
                    2 * alpha * beta)
    
    return grad, hess
    
def custom_loss_low(y_pred, dataset):
    y_true = dataset.get_label()
    residual = (y_true - y_pred).astype("float")

    # 定义损失函数参数
    alpha = 10.0  # 控制惩罚力度
    beta = 1.0    # 控制损失函数的非对称性

    # 计算梯度：对于残差（真实值-预测值）为负的情况，施加更大的惩罚
    grad = np.where(residual < 0, 
                    -2 * alpha * (1 - beta) * residual,  # 预测值大于或等于真实值时
                    -2 * alpha * beta * residual)        # 预测值小于真实值时
    
    # 计算二阶导数：同样区分正负残差
    hess = np.where(residual < 0,
                    2 * alpha * (1 - beta),  # 预测值大于或等于真实值时
                    2 * alpha * beta)        # 预测值小于真实值时

    return grad, hess

# 定义参数
lgb_params = {
    'objective': custom_loss,  # 使用自定义损失函数作为目标函数
    'metric': 'rmse',           # 使用 RMSE 作为评估指标
}
lgb_params_low = {
    'objective': custom_loss_low,  # 使用自定义损失函数作为目标函数
    'metric': 'rmse',           # 使用 RMSE 作为评估指标
}

# LightGBM base model
#lgb_model = lgb.LGBMRegressor(**lgb_params)

# 使用 MultiOutputRegressor 包装带有自定义损失函数的 LightGBM 模型
#multioutput_model = MultiOutputRegressor(
#    lgb_model.set_params(**{'fobj': custom_loss})
#)

# 生成示例数据
data = pd.read_csv(f'{PATH}/data/test_603893.csv')
#data['high'] = data['high'] / data['close']
columns_to_divide = ['high', 'low', 'open', 'close']
data[columns_to_divide] = data[columns_to_divide].div(data['preclose'], axis=0)

data[['next_high', 'next_low']] = data[['high', 'low']].shift(-1)
#data['next_close'] = data['close'].shift(-1)
data = data.head(-1)

X = data[['open', 'high', 'low', 'close', 'volume',
       'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM',
       'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']]
y = data[['next_high', 'next_low']]

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 现在你可以像普通模型一样进行 fit 和 predict
#multioutput_model.fit(X_train, y_train)
#predictions = multioutput_model.predict(X_test)
# 多输出回归 - 手动遍历多个目标
def multioutput_train(X_train, y_train, X_test, y_test):
    models = []
    predictions = []
    
    # 假设 y_train 是 pandas.DataFrame 或 numpy.ndarray
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()  # 将 DataFrame 转换为 numpy 数组
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy()    # 将 DataFrame 转换为 numpy 数组

    # 遍历每个输出（每一列是一个目标）
    for i in range(y_train.shape[1]):
        #print('i',i)
        print(f"Training model for output {i + 1}/{y_train.shape[1]}")
        
        # 创建 LightGBM 数据集
        lgb_train = lgb.Dataset(X_train, label=y_train[:, i])  # 训练数据第 i 个输出
        lgb_valid = lgb.Dataset(X_test, label=y_test[:, i], reference=lgb_train)  # 测试数据第 i 个输出
        
        if i==0:
            # 使用自定义损失函数进行训练
            model = lgb.train(
                params=lgb_params,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                #callbacks=[lgb.callback.PrintEvaluation(100)]  # 每 100 次迭代打印一次评估
            )
        if i==1:
            model = lgb.train(
                params=lgb_params_low,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                #callbacks=[lgb.callback.PrintEvaluation(100)]  # 每 100 次迭代打印一次评估
            )
        # 保存每个模型
        models.append(model)
        
        # 预测
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    # 将所有输出的预测结果合并
    predictions = np.column_stack(predictions)
    return models, predictions

# 训练模型
models, predictions = multioutput_train(X_train, y_train, X_test, y_test)


# 计算 MSE
mse = mean_squared_error(y_test.iloc[:,0], predictions[:,0])


result = pd.DataFrame(np.hstack((y_test.values, predictions)), columns=['next_high_real',
                                                                        'next_low_real',
                                                                        'next_high_pred',
                                                                        'next_low_pred',
                                                                        ])


result['next_high_bool'] = np.where(result['next_high_real'] >= result['next_high_pred'], 1, None)
result['next_low_bool'] = np.where(result['next_low_real'] <= result['next_low_pred'], 1, None)
print(result)
print(f"Mean Squared Error: {mse}")
print(result.sum())
#result.to_csv(f'{PATH}/data/test_result_multioutput.csv',index=False)
#Mean Squared Error: 0.0012499485110463058
# =============================================================================
# next_high_real    215.317793
# next_low_real     204.862184
# next_high_pred    214.265353
# next_low_pred     204.648853
# next_high_bool           107
# next_low_bool             91
# dtype: object
# =============================================================================
