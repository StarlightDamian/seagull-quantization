# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:25:25 2024

@author: awei
lightgbm_train_reg
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from __init__ import path
from utils import utils_database, utils_log
import lightgbm_base

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

def custom_loss(y_pred, y_true):
    #y_true = dataset.get_label()
    delta = y_true - y_pred
    alpha = 1.2  # Huber 损失的阈值参数，可以调整

    # 梯度和 Hessian
    grad = np.where(np.abs(delta) <= alpha, -delta, -alpha * np.sign(delta))
    hess = np.where(np.abs(delta) <= alpha, 0.9, 0.2)
    return grad, hess

if __name__ == '__main__':
    with utils_database.engine_conn('postgre') as conn:
        ohlc_df = pd.read_sql('das_wide_incr_train', con=conn.engine)
        # ohlc_df = pd.read_sql("select * from das_wide_incr_train where board_type='主板'", con=conn.engine)
        # ohlc_df = pd.read_sql("select *  from das_wide_incr_train where board_type not in ('北交所','ETF') ", con=conn.engine)
        
    # 清洗脏数据
    ohlc_df = ohlc_df[(ohlc_df.high <= ohlc_df.limit_up)&(ohlc_df.low >= ohlc_df.limit_down)]
    
    ohlc_features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume', 'turnover', 'turnover_pct',
                     'price_limit_rate','board_type','full_code', 'date_diff', 'date_week','is_flat_price_prev','is_limit_down_prev',
                     'is_limit_up_prev']
    fundamental_features = ['chg_rel', 'pe_ttm', 'ps_ttm', 'pcf_ttm', 'pb_mrq']
    # =============================================================================
    #     # 多标签进行onehot， 先置空加快计算
    #     ohlc_df = onehot(ohlc_df, label_df[['full_code', 'label']])
    #     label_features =  [x for x in ohlc_df.columns if ('label_' in x) and ('label_昨日' not in x)]#[]#
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
           'small_inflow_hist_diff_1',
           'main_small_net_inflow_slope_12_26_9',
           'main_small_net_inflow_acceleration_12_26_9',
           'main_small_net_inflow_hist_12_26_9',
           'main_small_net_inflow_diff_1',
           'main_small_net_inflow_diff_5',
           'main_small_net_inflow_diff_30',
           'main_small_net_inflow_hist_diff_1']
    alpha_features = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']
    index_features = ['index_close_diff_1', 'index_close_diff_5', 'index_close_diff_30', 'index_volume_diff_1','index_volume_diff_5', 'index_volume_diff_30', 'index_turnover_diff_1', 'index_turnover_diff_5', 'index_turnover_diff_30','index_close_rate']
    indicators_features = ['rsi','cci', 'wr', 'vwap', 'ad', 'mom', 'atr', 'adx', 'plus_di', 'minus_di', 'mfi', 'upper_band', 'middle_band', 'lower_band', 'kdj_fastk','kdj_fastd']
    
    features = ohlc_features +\
               fundamental_features +\
               label_features +\
               macd_features +\
               flow_features +\
               alpha_features +\
               index_features +\
               indicators_features
               
    # 特征和目标
    x = ohlc_df[features]
    y_reg = ohlc_df['next_high_rate']
    
    # 处理异常值
    y_reg = y_reg.replace([np.inf, -np.inf, np.nan], 0)
    x = x.replace([np.inf, -np.inf, np.nan], 1)  #.fillna(1)  # debug:这个错误是由于输入数据中包含无穷值 (inf)、负无穷值 (-inf)、或超出 float64 类型范围的值导致的
    constant_features = x.columns[x.nunique() == 1]  # 筛选出所有值相同的列
    logger.warning(f'相同值特征: {constant_features}')  # ['alpha019', 'alpha027', 'alpha039']
    x = x.loc[:, x.nunique() > 1]
    
    # 特征选择
    categorical_features = ['board_type','full_code','date_week']
    numeric_features = list(set(features)-set(constant_features)-set(categorical_features))
    # x[numeric_features] = x[numeric_features].astype(float)
    
    
    # 划分训练和测试集
    x_train, x_test, y_reg_train, y_reg_test = train_test_split(
        x, y_reg, test_size=0.2, random_state=42)
    
    # 回归
# =============================================================================
#     params = {
#         'task': 'train',
#         'boosting': 'gbdt',
#         'objective': 'regression',
#         'max_depth':9,
#         'num_leaves': 511,  # 决策树上的叶子节点的数量，控制树的复杂度
#         'learning_rate': 0.12,  # 0.05,0.1
#         'metric': ['root_mean_squared_error'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error',mae
#         #w×RMSE+(1−w)×MAE
#         'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
#         #'early_stop_round':20,
#         'n_estimators': 1500,
#         #'min_child_sample':40,
#         #'min_child_weight':1,
#         #'subsample':0.8,
#         #'colsample_bytree':0.8,
#     }
# =============================================================================
    param_grid_class = {
    'regressor__objective': ['regression'],  # 回归
    'regressor__boosting_type': ['gbdt'],  # 梯度提升决策树
    'regressor__metric': ['neg_mean_squared_error'],  # 多分类交叉熵
    'regressor__num_leaves': [255],  # 树的叶子数
    'regressor__max_depth': [8],  # 最大深度
    'regressor__learning_rate': [0.12],  # 学习率
    'regressor__n_estimators': [1200], # [1200],  # 树的数量
    # 'regressor__min_child_samples': [10, 20, 30],  # 每个叶子节点的最小样本数
    # 'regressor__subsample': [0.7, 0.8, 1.0],  # 样本采样比例
    # 'regressor__colsample_bytree': [0.7, 0.8, 1.0],  # 每棵树的特征采样比例
    }
    pipeline = lightgbm_base.StockModelPipeline()

    #pipeline.fit_reg_model(x_train, y_reg_train, numeric_features + ['cluster_label'], categorical_features, param_grid_reg)
    pipeline.fit_reg_model(
    x_train,
    y_reg_train,
    numeric_features,# + ['cluster_label'],
    categorical_features,
    param_grid=param_grid_class,
    
    #regressor=LGBMRegressor(**params)  # 将自定义参数传递到回归器
)
# =============================================================================
#     # 替换默认损失函数
#     pipeline.regressor_.set_params(
#     objective=custom_loss,
#     #feval=custom_metric  # 可选：加入自定义评估指标
# )
# =============================================================================
    y_reg_pred = pipeline.predict_reg(x_test)
    
    print("Regression RMSE:")
    print(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)).round(4))
    result_reg = pd.DataFrame([y_reg_test.values, y_reg_pred]).T
    result_reg.columns=['reg_real','reg_pred']
    print(result_reg.head(25))
    result_reg.reg_real.mean()
    result_reg.reg_pred.mean()